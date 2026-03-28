# src/eval_ooc_object.py
#eval tren val2014+test2014
#logit DUONG (file cu AM)

import os
import random
from typing import List, Tuple, Dict
from datetime import datetime
import time

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from src.utils.io import load_config, ensure_dir
from src.utils.seed import seed_all
from src.utils.logger import Logger

from src.data.coco_instances import CocoInstances
from src.data.coco_ooc import CocoOOCDataset

from src.models.ijepa_backbone import IJepaBackbone
from src.models.detector_head import OOCDetectorHead


# =========================
# Utils: boxes + metrics
# =========================
def list_images_from_dirs(dirs: List[str]) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: List[str] = []
    for d in dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
    paths.sort()
    return paths


def xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = b
    return [x, y, x + w, y + h]


def xyxy_to_xywh(b: List[float]) -> List[float]:
    x0, y0, x1, y1 = b
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray):
    # sort by score desc
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    distinct = np.where(np.diff(y_score))[0]
    thr_idx = np.r_[distinct, y_true.size - 1]

    tps = np.cumsum(y_true)[thr_idx]
    fps = (1 + thr_idx) - tps

    P = np.sum(y_true)
    N = y_true.size - P

    if P == 0 or N == 0:
        fpr = np.array([0.0, 1.0], dtype=np.float64)
        tpr = np.array([0.0, 1.0], dtype=np.float64)
        thr = np.array([np.inf, -np.inf], dtype=np.float64)
        return fpr, tpr, thr

    tpr = tps / P
    fpr = fps / N
    thr = y_score[thr_idx]

    # prepend (0,0)
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thr = np.r_[thr[0] + 1e-12, thr]
    return fpr, tpr, thr


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    # np.trapz is deprecated; keep for compatibility on your env
    return float(np.trapz(y, x))


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve_np(y_true, y_score)
    return auc_trapz(fpr, tpr)


def average_precision_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    P = np.sum(y_true)
    if P == 0:
        return 0.0
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / P
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve_np(y_true, y_score)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(np.min(fpr[idx]))


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve_np(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.max(tpr[idx]))


def compute_object_metrics(y_true: List[int], y_score: List[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float64)
    return {
        "Object_AUROC": roc_auc_np(yt, ys),
        "Object_AUPRC": average_precision_np(yt, ys),
        "FPR@95TPR": fpr_at_tpr(yt, ys, 0.95),
        "TPR@5%FPR": tpr_at_fpr(yt, ys, 0.05),
        "NumPos": float(np.sum(yt)),
        "NumNeg": float(len(yt) - np.sum(yt)),
    }


# =========================
# Utils: crop + scoring
# =========================

def safe_crop_rgb(pil_img: Image.Image, bbox_xywh, min_size: int = 16) -> Image.Image:
    pil_img = pil_img.convert("RGB")
    x, y, w, h = bbox_xywh
    x0 = int(max(0, x))
    y0 = int(max(0, y))
    x1 = int(min(pil_img.width, x + w))
    y1 = int(min(pil_img.height, y + h))

    if x1 <= x0 or y1 <= y0:
        return pil_img

    crop = pil_img.crop((x0, y0, x1, y1)).convert("RGB")

    if crop.width < min_size or crop.height < min_size:
        new_w = max(min_size, crop.width)
        new_h = max(min_size, crop.height)
        crop = crop.resize((new_w, new_h), resample=Image.BILINEAR).convert("RGB")

    return crop


@torch.no_grad()
def score_box(
    backbone: IJepaBackbone,
    head: OOCDetectorHead,
    pil_img: Image.Image,
    bbox_xywh: List[float],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    min_crop_size: int = 16,
) -> float:
    """
    IMPORTANT: This returns an OOC score (higher => more OOC).
    We use: s_ooc = sigmoid(logit), assuming the head logit behaves like OOC.
    """
    pil_img = pil_img.convert("RGB")

    # ctx embedding
    ctx = backbone.processor(images=pil_img, return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        ctx_emb = backbone.encode(ctx)  # (1, D)

    # obj embedding
    obj_img = safe_crop_rgb(pil_img, bbox_xywh, min_size=min_crop_size)
    try:
        obj = backbone.processor(images=obj_img, return_tensors="pt")["pixel_values"].to(device)
    except Exception:
        obj = backbone.processor(images=pil_img, return_tensors="pt")["pixel_values"].to(device)

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        obj_emb = backbone.encode(obj)
        logit = head(obj_emb, ctx_emb)
        #prob_ooc = torch.sigmoid(-logit).item()  # <-- OOC-score
        prob_ooc = torch.sigmoid(logit).item()   # <-- OOC-score (Positive=OOC)

    return float(prob_ooc)


# =========================
# Pred-box detector (torchvision)
# =========================

def build_torchvision_detector(name: str, device: torch.device):
    """
    name:
      - maskrcnn_resnet50_fpn
      - fasterrcnn_resnet50_fpn
    """
    import torchvision
    name = name.lower()
    if name == "maskrcnn_resnet50_fpn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    elif name == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        raise ValueError(f"Unsupported detector: {name}")

    model.eval().to(device)
    return model


@torch.no_grad()
def detector_predict_boxes_xyxy(
    det_model,
    pil_img: Image.Image,
    device: torch.device,
    det_conf: float,
    topn: int,
) -> Tuple[List[List[float]], List[float]]:
    """
    Returns filtered (boxes_xyxy, det_scores).
    """
    import torchvision.transforms.functional as TF

    img = pil_img.convert("RGB")
    x = TF.to_tensor(img).to(device)
    out = det_model([x])[0]

    boxes = out["boxes"].detach().cpu().numpy().tolist()  # xyxy
    scores = out["scores"].detach().cpu().numpy().tolist()

    keep = [(b, s) for b, s in zip(boxes, scores) if float(s) >= float(det_conf)]
    keep.sort(key=lambda t: t[1], reverse=True)
    if topn is not None and topn > 0:
        keep = keep[: int(topn)]

    boxes_xyxy = [b for b, _ in keep]
    scores_det = [float(s) for _, s in keep]
    return boxes_xyxy, scores_det


# =========================
# Main
# =========================

def main(cfg_path: str, run_dir_override: str = ""):
    cfg = load_config(cfg_path)

    seed = int(cfg["project"]["seed"])
    seed_all(seed)
    random.seed(seed)

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["runtime"]["amp"])
    amp_dtype = torch.bfloat16 if str(cfg["runtime"]["amp_dtype"]).lower() == "bf16" else torch.float16

    # ---- run_dir: default from cfg OR override from CLI ----
    out_root = cfg["project"]["out_dir"]
    ensure_dir(out_root)
    default_run_dir = os.path.join(out_root, cfg["project"]["name"])

    if run_dir_override:
        run_dir = os.path.abspath(os.path.expanduser(run_dir_override))
    else:
        run_dir = default_run_dir

    if not os.path.exists(run_dir):
        ensure_dir(run_dir)

    ckpt_dir = cfg["project"].get("ckpt_dir", "")
    if ckpt_dir:
        ckpt_dir = os.path.abspath(os.path.expanduser(ckpt_dir))
    else:
        ckpt_dir = run_dir

    logger = Logger(run_dir, "eval_ooc_object")
    eval_t0 = time.time()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[Eval] Start time: {start_ts}")
    out_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.log(f"[Paths] run_dir={run_dir} (default would be {default_run_dir})")
    logger.log(f"[Paths] ckpt_dir={ckpt_dir}")
    logger.log(f"[Paths] log_file={logger.path}")
    logger.log("SCORE_MODE = sigmoid(logit)  [OOC-score: higher => more OOC]")

    # ---- object-eval config ----
    obj_cfg = cfg.get("eval_object", {})
    mode = str(obj_cfg.get("mode", "both")).lower()  # oracle | pred | both
    include_normal = bool(obj_cfg.get("include_normal", True))
    max_ooc = int(obj_cfg.get("max_ooc_images", 106036))
    max_normal = int(obj_cfg.get("max_normal_images", -1))
    min_crop = int(cfg["image"].get("min_crop_size", 16))

    # Pred-box
    det_name = str(obj_cfg.get("detector", "maskrcnn_resnet50_fpn"))
    det_conf = float(obj_cfg.get("det_conf", 0.10))  # LOW => higher recall
    topn = int(obj_cfg.get("topn", 100))             # HIGH => higher recall
    iou_thr = float(obj_cfg.get("iou_thr", 0.50))
    miss_mode = str(obj_cfg.get("miss_mode", "conservative")).lower()  # conservative | ignore

    logger.log(f"[ObjectEval] mode={mode} include_normal={include_normal} max_ooc={max_ooc} max_normal={max_normal} min_crop={min_crop}")
    logger.log(f"[PredBoxes] detector={det_name} det_conf={det_conf} topn={topn} iou_thr={iou_thr} miss_mode={miss_mode}")

    # ---- Load COCO instances ----
    coco_val = CocoInstances(cfg["paths"]["instances_val"])
    val_ids = list(coco_val.img_by_id.keys())

    normal_dirs = cfg.get("paths", {}).get("normal_dirs", [])
    if isinstance(normal_dirs, str):
        normal_dirs = [normal_dirs]
    normal_paths = list_images_from_dirs(normal_dirs) if normal_dirs else []
    if max_normal > 0:
        val_ids = val_ids[:max_normal]
        if normal_paths:
            normal_paths = normal_paths[:max_normal]
    if include_normal:
        if normal_paths:
            logger.log(f"[ObjectEval] Normal images used (normal_dirs): {len(normal_paths)}")
        else:
            logger.log(f"[ObjectEval] Normal images used (COCO val): {len(val_ids)}")

    # ---- Build backbone + load checkpoints ----
    backbone = IJepaBackbone(cfg["hf"]["model_id"], attn_implementation=cfg["hf"].get("attn_implementation", "sdpa"))
    backbone.to(device)
    backbone.model.eval()

    ssl_ckpt = os.path.join(ckpt_dir, "ssl_final.pt")
    logger.log(f"[Paths] ssl_ckpt={ssl_ckpt}")
    if os.path.exists(ssl_ckpt):
        sd = torch.load(ssl_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "student" in sd:
            backbone.model.load_state_dict(sd["student"], strict=False)
        else:
            backbone.model.load_state_dict(sd, strict=False)
        logger.log(f"Loaded SSL backbone from {ssl_ckpt}")
    else:
        logger.log("No SSL checkpoint found. Using pure HF pretrained backbone.")

    det_ckpt = os.path.join(ckpt_dir, "detector_final.pt")
    if not os.path.exists(det_ckpt):
        det_ckpt = os.path.join(ckpt_dir, "detector_head.pt")
    logger.log(f"[Paths] det_ckpt={det_ckpt}")
    if not os.path.exists(det_ckpt):
        raise FileNotFoundError(f"Missing detector checkpoint in {ckpt_dir} (expected detector_final.pt or detector_head.pt)")

    det_state = torch.load(det_ckpt, map_location="cpu")
    logger.log(f"Loaded detector head checkpoint from {det_ckpt}")

    # infer embedding dim
    with torch.no_grad():
        dummy = torch.zeros((1, 3, cfg["image"]["size"], cfg["image"]["size"]), device=device)
        dim = backbone.encode(dummy).shape[-1]

    head = OOCDetectorHead(
        dim=dim,
        hidden=int(cfg["detector"]["head_hidden"]),
        dropout=float(cfg["detector"]["dropout"]),
    )
    if isinstance(det_state, dict) and "head" in det_state:
        head.load_state_dict(det_state["head"])
    else:
        head.load_state_dict(det_state)
    head.to(device).eval()

    # ---- Create OOC dataset (needs processor) ----
    ooc_ds = CocoOOCDataset(
        cfg["paths"]["ooc_images"],
        cfg["paths"]["ooc_ann_dir"],
        processor=backbone.processor,
        max_items=max_ooc,
    )
    logger.log(f"[ObjectEval] OOC images used: {len(ooc_ds)}")

    # =========================
    # ORACLE (GT boxes)
    # =========================
    if mode in ("oracle", "both"):
        logger.log("=== ORACLE (GT boxes) Object-level evaluation ===")
        y_true: List[int] = []
        y_score: List[float] = []
        img_idx_list: List[int] = []
        role_list: List[int] = []  # 0=neg/bg, 1=pos/ooc

        # Normal: all GT boxes negative
        if include_normal:
            if normal_paths:
                logger.log(f"[ORACLE] normal_dirs provided ({len(normal_paths)} images), but GT boxes require COCO val annotations. Using COCO val GT boxes instead.")
            logger.log(f"[ORACLE] Normal images used (COCO val): {len(val_ids)}")
            pbar = tqdm(val_ids, desc="oracle normal (GT boxes)", dynamic_ncols=True)
            for image_id in pbar:
                fn = coco_val.image_file(image_id)
                path = os.path.join(cfg["paths"]["val_images"], fn)
                pil_img = Image.open(path).convert("RGB")

                anns = coco_val.anns_for_image(image_id)
                for a in anns:
                    s = score_box(backbone, head, pil_img, a["bbox"], device, use_amp, amp_dtype, min_crop_size=min_crop)
                    y_true.append(0)
                    y_score.append(s)
                    img_idx_list.append(-1)  # normal split marker
                    role_list.append(0)

        # OOC: bg GT boxes negative + OOC GT box positive
        pbar = tqdm(range(len(ooc_ds)), desc="oracle ooc (GT boxes)", dynamic_ncols=True)
        for i in pbar:
            item = ooc_ds[i]
            entry = item["entry"]
            pil_img = Image.open(item["image_path"]).convert("RGB")

            # bg objects (neg)
            for aid in entry["original_ann_ids"]:
                a = coco_val.ann(aid)
                if a is None:
                    continue
                s = score_box(backbone, head, pil_img, a["bbox"], device, use_amp, amp_dtype, min_crop_size=min_crop)
                y_true.append(0)
                y_score.append(s)
                img_idx_list.append(i)
                role_list.append(0)

            # ooc object (pos)
            ooc_box = entry["ooc_annotation"]["bbox"]
            s_pos = score_box(backbone, head, pil_img, ooc_box, device, use_amp, amp_dtype, min_crop_size=min_crop)
            y_true.append(1)
            y_score.append(s_pos)
            img_idx_list.append(i)
            role_list.append(1)

        m = compute_object_metrics(y_true, y_score)
        logger.log(f"[ORACLE] {m}")

        out_npz = os.path.join(run_dir, f"object_oracle_scores_{out_ts}.npz")
        logger.log(f"[Paths] oracle_scores_out={out_npz}")
        np.savez_compressed(
            out_npz,
            y_true=np.asarray(y_true),
            y_score=np.asarray(y_score),
            img_idx=np.asarray(img_idx_list, dtype=np.int32),
            role=np.asarray(role_list, dtype=np.int8),
        )
        logger.log(f"[ORACLE] Saved raw scores to {out_npz}")

    # =========================
    # PRED (detector boxes)
    # =========================
    if mode in ("pred", "both"):
        logger.log("=== PRED (detector boxes) Object-level evaluation ===")
        det_model = build_torchvision_detector(det_name, device)

        y_true: List[int] = []
        y_score: List[float] = []
        img_idx_list: List[int] = []
        role_list: List[int] = []  # 0=neg(det box), 1=pos(best-match), 2=miss-pos

        ooc_hits = 0
        ooc_total = 0

        # Normal: all predicted boxes negative
        if include_normal:
            if normal_paths:
                logger.log(f"[PRED] Normal images used (normal_dirs): {len(normal_paths)}")
                pbar = tqdm(normal_paths, desc="pred normal (dir images)", dynamic_ncols=True)
                for path in pbar:
                    pil_img = Image.open(path).convert("RGB")
                    boxes_xyxy, _ = detector_predict_boxes_xyxy(det_model, pil_img, device, det_conf=det_conf, topn=topn)
                    for bxyxy in boxes_xyxy:
                        s = score_box(backbone, head, pil_img, xyxy_to_xywh(bxyxy), device, use_amp, amp_dtype, min_crop_size=min_crop)
                        y_true.append(0)
                        y_score.append(s)
                        img_idx_list.append(-1)
                        role_list.append(0)
            else:
                logger.log(f"[PRED] Normal images used (COCO val): {len(val_ids)}")
                pbar = tqdm(val_ids, desc="pred normal (det boxes)", dynamic_ncols=True)
                for image_id in pbar:
                    fn = coco_val.image_file(image_id)
                    path = os.path.join(cfg["paths"]["val_images"], fn)
                    pil_img = Image.open(path).convert("RGB")

                    boxes_xyxy, _ = detector_predict_boxes_xyxy(det_model, pil_img, device, det_conf=det_conf, topn=topn)
                    for bxyxy in boxes_xyxy:
                        s = score_box(backbone, head, pil_img, xyxy_to_xywh(bxyxy), device, use_amp, amp_dtype, min_crop_size=min_crop)
                        y_true.append(0)
                        y_score.append(s)
                        img_idx_list.append(-1)
                        role_list.append(0)

        # OOC: best-match labeling
        logger.log(f"[PRED] OOC images used: {len(ooc_ds)}")
        pbar = tqdm(range(len(ooc_ds)), desc="pred ooc (det boxes)", dynamic_ncols=True)
        for i in pbar:
            item = ooc_ds[i]
            entry = item["entry"]
            pil_img = Image.open(item["image_path"]).convert("RGB")

            boxes_xyxy, _ = detector_predict_boxes_xyxy(det_model, pil_img, device, det_conf=det_conf, topn=topn)

            ooc_total += 1
            ooc_gt_xyxy = xywh_to_xyxy(entry["ooc_annotation"]["bbox"])

            if len(boxes_xyxy) == 0:
                # detector outputs nothing => miss
                if miss_mode == "conservative":
                    # Since score is OOC-score (higher => more OOC), a conservative miss should be high.
                    y_true.append(1)
                    y_score.append(1.0)
                    img_idx_list.append(i)
                    role_list.append(2)  # miss-pos
                continue

            ious = [box_iou_xyxy(b, ooc_gt_xyxy) for b in boxes_xyxy]
            best_idx = int(np.argmax(np.asarray(ious)))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_thr:
                ooc_hits += 1

                for j, bxyxy in enumerate(boxes_xyxy):
                    s = score_box(backbone, head, pil_img, xyxy_to_xywh(bxyxy), device, use_amp, amp_dtype, min_crop_size=min_crop)
                    if j == best_idx:
                        y_true.append(1)
                        y_score.append(s)
                        img_idx_list.append(i)
                        role_list.append(1)  # best-match pos
                    else:
                        y_true.append(0)
                        y_score.append(s)
                        img_idx_list.append(i)
                        role_list.append(0)
            else:
                # miss
                if miss_mode == "conservative":
                    y_true.append(1)
                    y_score.append(1.0)  # conservative for OOC-score
                    img_idx_list.append(i)
                    role_list.append(2)

                # still keep predicted boxes as negatives (reflects real pipeline)
                for bxyxy in boxes_xyxy:
                    s = score_box(backbone, head, pil_img, xyxy_to_xywh(bxyxy), device, use_amp, amp_dtype, min_crop_size=min_crop)
                    y_true.append(0)
                    y_score.append(s)
                    img_idx_list.append(i)
                    role_list.append(0)

        recall_det = ooc_hits / max(1, ooc_total)
        m = compute_object_metrics(y_true, y_score)
        m[f"Recall_det@IoU{float(iou_thr):.2f}"] = float(recall_det)
        m["OOC_total"] = int(ooc_total)
        m["OOC_hits"] = int(ooc_hits)

        logger.log(f"[PRED] {m}")

        out_npz = os.path.join(run_dir, f"object_pred_scores_{out_ts}.npz")
        logger.log(f"[Paths] pred_scores_out={out_npz}")
        np.savez_compressed(
            out_npz,
            y_true=np.asarray(y_true),
            y_score=np.asarray(y_score),
            img_idx=np.asarray(img_idx_list, dtype=np.int32),
            role=np.asarray(role_list, dtype=np.int8),
        )
        logger.log(f"[PRED] Saved raw scores to {out_npz}")

    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - eval_t0
    logger.log(f"[Eval] End time: {end_ts}")
    logger.log(f"[Eval] Elapsed time: {elapsed:.1f}s")
    logger.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
   
    args = ap.parse_args()
    run_dir_override = getattr(args, "run_dir", "")
    main(args.cfg, run_dir_override=run_dir_override)
