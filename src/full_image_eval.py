# src/full_image_eval.py
import os
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from src.utils.io import load_config, ensure_dir
from src.utils.seed import seed_all
from src.utils.logger import Logger
from src.utils.metrics import compute_metrics

from src.data.coco_instances import CocoInstances
from src.data.coco_ooc import CocoOOCDataset

from src.models.ijepa_backbone import IJepaBackbone
from src.models.detector_head import OOCDetectorHead


# -------------------------
# Detector: predicted boxes
# -------------------------
def build_torchvision_detector(name: str, device: torch.device):
    """
    Supported:
      - maskrcnn_resnet50_fpn
      - fasterrcnn_resnet50_fpn
    """
    import torchvision

    name = str(name).lower()
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
    Returns:
      boxes_xyxy: list of [x0,y0,x1,y1]
      scores_det: list of detector scores
    """
    import torchvision.transforms.functional as TF

    img = pil_img.convert("RGB")
    x = TF.to_tensor(img).to(device)
    out = det_model([x])[0]

    boxes = out["boxes"].detach().cpu().numpy().tolist()
    scores = out["scores"].detach().cpu().numpy().tolist()

    keep = [(b, s) for b, s in zip(boxes, scores) if float(s) >= float(det_conf)]
    keep.sort(key=lambda t: float(t[1]), reverse=True)
    if topn is not None and int(topn) > 0:
        keep = keep[: int(topn)]

    boxes_xyxy = [b for b, _ in keep]
    scores_det = [float(s) for _, s in keep]
    return boxes_xyxy, scores_det


def xyxy_to_xywh(bxyxy: List[float]) -> List[float]:
    x0, y0, x1, y1 = bxyxy
    return [float(x0), float(y0), float(max(0.0, x1 - x0)), float(max(0.0, y1 - y0))]


# -------------------------
# Crop + scoring
# -------------------------
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
        crop = crop.resize(
            (max(min_size, crop.width), max(min_size, crop.height)),
            resample=Image.BILINEAR,
        ).convert("RGB")
    return crop


@torch.no_grad()
def score_boxes_object_context(
    backbone: IJepaBackbone,
    head: OOCDetectorHead,
    pil_img: Image.Image,
    boxes_xywh: List[List[float]],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    min_crop: int = 16,
    # IMPORTANT: head_direction
    # - if your head outputs "compatibility" prob -> OOC prob = sigmoid(-logit)
    # - if your head outputs "ooc" prob -> OOC prob = sigmoid(logit)
    use_sigmoid_neg_logit: bool = False,
) -> List[float]:
    """
    Returns object-level scores (one per box).
    """
    pil_img = pil_img.convert("RGB")

    # context embedding once
    ctx = backbone.processor(images=pil_img, return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        ctx_emb = backbone.encode(ctx)  # (1, D)

    scores: List[float] = []
    for b in boxes_xywh:
        obj_img = safe_crop_rgb(pil_img, b, min_size=min_crop)
        obj = backbone.processor(images=obj_img, return_tensors="pt")["pixel_values"].to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            obj_emb = backbone.encode(obj)
            logit = head(obj_emb, ctx_emb)  # (1,)
            if use_sigmoid_neg_logit:
                s = torch.sigmoid(-logit).item()
            else:
                s = torch.sigmoid(logit).item()

        scores.append(float(s))

    return scores


def aggregate_image_score(obj_scores: List[float], score_mode: str = "topk_mean", topk: int = 3) -> float:
    if len(obj_scores) == 0:
        return 0.0

    score_mode = str(score_mode).lower()
    if score_mode == "max":
        return float(np.max(np.asarray(obj_scores, dtype=np.float32)))
    # default: topk_mean
    s = sorted(obj_scores, reverse=True)
    k = max(1, min(int(topk), len(s)))
    return float(np.mean(s[:k]))


# -------------------------
# Main full eval
# -------------------------
def main(cfg_path: str, run_dir: str, auto_check_n: int = 2000):
    cfg = load_config(cfg_path)
    seed = int(cfg["project"]["seed"])
    seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["runtime"]["amp"])
    amp_dtype = torch.bfloat16 if str(cfg["runtime"]["amp_dtype"]).lower() == "bf16" else torch.float16
    min_crop = int(cfg["image"].get("min_crop_size", 16))

    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    ensure_dir(run_dir)
    logger = Logger(run_dir, "full_image_eval")
    logger.log(f"[Paths] cfg={cfg_path} run_dir={run_dir}")

    # ---- eval config (image-level) ----
    eval_cfg = cfg.get("eval", {})
    max_ooc = int(eval_cfg.get("max_ooc_images", -1))
    score_mode = str(eval_cfg.get("score_mode", "topk_mean"))
    topk = int(eval_cfg.get("topk", 3))

    # detector params (prefer eval_cfg; fallback to eval_object if you used that)
    eval_obj_cfg = cfg.get("eval_object", {})
    det_name = str(eval_cfg.get("detector", eval_obj_cfg.get("detector", "maskrcnn_resnet50_fpn")))
    det_conf = float(eval_cfg.get("det_conf", eval_obj_cfg.get("det_conf", 0.10)))
    topn = int(eval_cfg.get("topn", eval_obj_cfg.get("topn", 300)))

    logger.log(f"[Eval] max_ooc_images={max_ooc} score_mode={score_mode} topk={topk}")
    logger.log(f"[Detector] name={det_name} det_conf={det_conf} topn={topn}")

    # ---- load COCO val instances (normal images) ----
    coco = CocoInstances(cfg["paths"]["instances_val"])
    val_ids = list(coco.img_by_id.keys())
    logger.log(f"[Data] COCO val images (normal candidates) = {len(val_ids)}")

    # ---- build backbone + load ssl ----
    backbone = IJepaBackbone(cfg["hf"]["model_id"], attn_implementation=cfg["hf"].get("attn_implementation", "sdpa"))
    backbone.to(device)
    backbone.model.eval()

    ssl_ckpt = os.path.join(run_dir, "ssl_final.pt")
    if os.path.exists(ssl_ckpt):
        sd = torch.load(ssl_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "student" in sd:
            backbone.model.load_state_dict(sd["student"], strict=False)
        else:
            backbone.model.load_state_dict(sd, strict=False)
        logger.log(f"Loaded SSL backbone from {ssl_ckpt}")
    else:
        logger.log("No ssl_final.pt found. Using HF pretrained backbone.")

    # infer dim
    with torch.no_grad():
        dummy = torch.zeros((1, 3, cfg["image"]["size"], cfg["image"]["size"]), device=device)
        dim = backbone.encode(dummy).shape[-1]

    # ---- load detector head ----
    det_ckpt = os.path.join(run_dir, "detector_final.pt")
    if not os.path.exists(det_ckpt):
        det_ckpt = os.path.join(run_dir, "detector_head.pt")
    if not os.path.exists(det_ckpt):
        raise FileNotFoundError(f"Missing detector checkpoint in {run_dir} (detector_final.pt or detector_head.pt)")

    det_state = torch.load(det_ckpt, map_location="cpu")
    head = OOCDetectorHead(dim=dim, hidden=int(cfg["detector"]["head_hidden"]), dropout=float(cfg["detector"]["dropout"]))
    head.load_state_dict(det_state["head"] if isinstance(det_state, dict) and "head" in det_state else det_state)
    head.to(device)
    head.eval()
    logger.log(f"Loaded detector head from {det_ckpt}")

    # ---- build torchvision detector for predicted boxes ----
    det_model = build_torchvision_detector(det_name, device)

    # ---- load OOC dataset (composite images) ----
    ooc_ds = CocoOOCDataset(
        cfg["paths"]["ooc_images"],
        cfg["paths"]["ooc_ann_dir"],
        processor=backbone.processor,
        max_items=max_ooc if max_ooc > 0 else -1,
    )
    logger.log(f"[Data] OOC images = {len(ooc_ds)}")

    # -------------------------------------------------------
    # Step A: auto check score direction on a subset
    # -------------------------------------------------------
    # We try both:
    #   A) use_sigmoid_neg_logit=False (raw sigmoid(logit))
    #   B) use_sigmoid_neg_logit=True  (sigmoid(-logit))
    # Choose the one giving higher AUROC on subset.
    #
    # This is safer than assuming the direction.
    # -------------------------------------------------------
    def collect_subset_scores(n_each: int = 1000, use_sigmoid_neg_logit: bool = False):
        y_true_sub: List[int] = []
        y_score_sub: List[float] = []

        # sample normal subset
        ids = val_ids.copy()
        random.shuffle(ids)
        ids = ids[:n_each]

        for image_id in tqdm(ids, desc=f"auto-check normal ({n_each})", dynamic_ncols=True):
            fn = coco.image_file(image_id)
            img_path = os.path.join(cfg["paths"]["val_images"], fn)
            pil_img = Image.open(img_path).convert("RGB")

            boxes_xyxy, _ = detector_predict_boxes_xyxy(det_model, pil_img, device, det_conf=det_conf, topn=topn)
            boxes_xywh = [xyxy_to_xywh(b) for b in boxes_xyxy]
            obj_scores = score_boxes_object_context(
                backbone, head, pil_img, boxes_xywh, device, use_amp, amp_dtype, min_crop=min_crop,
                use_sigmoid_neg_logit=use_sigmoid_neg_logit
            )
            s_img = aggregate_image_score(obj_scores, score_mode=score_mode, topk=topk)
            y_true_sub.append(0)
            y_score_sub.append(s_img)

        # sample ooc subset
        m = min(n_each, len(ooc_ds))
        idxs = list(range(len(ooc_ds)))
        random.shuffle(idxs)
        idxs = idxs[:m]

        for i in tqdm(idxs, desc=f"auto-check ooc ({m})", dynamic_ncols=True):
            item = ooc_ds[i]
            pil_img = Image.open(item["image_path"]).convert("RGB")

            boxes_xyxy, _ = detector_predict_boxes_xyxy(det_model, pil_img, device, det_conf=det_conf, topn=topn)
            boxes_xywh = [xyxy_to_xywh(b) for b in boxes_xyxy]
            obj_scores = score_boxes_object_context(
                backbone, head, pil_img, boxes_xywh, device, use_amp, amp_dtype, min_crop=min_crop,
                use_sigmoid_neg_logit=use_sigmoid_neg_logit
            )
            s_img = aggregate_image_score(obj_scores, score_mode=score_mode, topk=topk)
            y_true_sub.append(1)
            y_score_sub.append(s_img)

        return y_true_sub, y_score_sub

    n_each = max(200, min(1000, auto_check_n // 2))
    logger.log(f"[AutoCheck] Using subset n_normal={n_each}, n_ooc={n_each} to choose score direction")

    yt_a, ys_a = collect_subset_scores(n_each=n_each, use_sigmoid_neg_logit=False)
    m_a = compute_metrics(yt_a, ys_a)

    yt_b, ys_b = collect_subset_scores(n_each=n_each, use_sigmoid_neg_logit=True)
    m_b = compute_metrics(yt_b, ys_b)

    logger.log(f"[AutoCheck] sigmoid(logit) metrics: {m_a}")
    logger.log(f"[AutoCheck] sigmoid(-logit) metrics: {m_b}")

    use_sigmoid_neg_logit = bool(m_b["AUROC"] > m_a["AUROC"])
    logger.log(f"[AutoCheck] Selected use_sigmoid_neg_logit={use_sigmoid_neg_logit}")

    # -------------------------------------------------------
    # Step B: full eval
    # -------------------------------------------------------
    y_true: List[int] = []
    y_score_raw: List[float] = []  # the chosen direction score (already OOC score)
    image_paths: List[str] = []

    # 1) normal images (COCO val)
    logger.log("=== FULL IMAGE-LEVEL EVAL: NORMAL ===")
    pbar = tqdm(val_ids, desc="full normal", dynamic_ncols=True)
    for image_id in pbar:
        fn = coco.image_file(image_id)
        img_path = os.path.join(cfg["paths"]["val_images"], fn)
        pil_img = Image.open(img_path).convert("RGB")

        boxes_xyxy, _ = detector_predict_boxes_xyxy(det_model, pil_img, device, det_conf=det_conf, topn=topn)
        boxes_xywh = [xyxy_to_xywh(b) for b in boxes_xyxy]
        obj_scores = score_boxes_object_context(
            backbone, head, pil_img, boxes_xywh, device, use_amp, amp_dtype, min_crop=min_crop,
            use_sigmoid_neg_logit=use_sigmoid_neg_logit
        )
        s_img = aggregate_image_score(obj_scores, score_mode=score_mode, topk=topk)

        y_true.append(0)
        y_score_raw.append(float(s_img))
        image_paths.append(img_path)

    # 2) OOC images (COCO-OOC composites)
    logger.log("=== FULL IMAGE-LEVEL EVAL: OOC ===")
    pbar = tqdm(range(len(ooc_ds)), desc="full ooc", dynamic_ncols=True)
    for i in pbar:
        item = ooc_ds[i]
        img_path = item["image_path"]
        pil_img = Image.open(img_path).convert("RGB")

        boxes_xyxy, _ = detector_predict_boxes_xyxy(det_model, pil_img, device, det_conf=det_conf, topn=topn)
        boxes_xywh = [xyxy_to_xywh(b) for b in boxes_xyxy]
        obj_scores = score_boxes_object_context(
            backbone, head, pil_img, boxes_xywh, device, use_amp, amp_dtype, min_crop=min_crop,
            use_sigmoid_neg_logit=use_sigmoid_neg_logit
        )
        s_img = aggregate_image_score(obj_scores, score_mode=score_mode, topk=topk)

        y_true.append(1)
        y_score_raw.append(float(s_img))
        image_paths.append(img_path)

    # compute metrics
    metrics = compute_metrics(y_true, y_score_raw)
    logger.log(f"[FULL] {metrics}")

    # save
    out_npz = os.path.join(run_dir, "image_scores.npz")
    np.savez_compressed(
        out_npz,
        y_true=np.asarray(y_true, dtype=np.int32),
        y_score=np.asarray(y_score_raw, dtype=np.float32),
        image_paths=np.asarray(image_paths),
        meta=np.asarray(
            {
                "detector": det_name,
                "det_conf": det_conf,
                "topn": topn,
                "score_mode": score_mode,
                "topk": topk,
                "use_sigmoid_neg_logit": use_sigmoid_neg_logit,
                "max_ooc_images": max_ooc,
                "num_normal": len(val_ids),
                "num_ooc": len(ooc_ds),
                "run_dir": run_dir,
            },
            dtype=object,
        ),
    )
    logger.log(f"[FULL] Saved raw scores to {out_npz}")

    logger.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--run_dir", default="./runs/ijepa_coco2014_ooc_paste")
    ap.add_argument(
        "--auto_check_n",
        type=int,
        default=2000,
        help="Number of images used to auto-select score direction (split half normal/half ooc).",
    )
    args = ap.parse_args()

    # IMPORTANT: use module-style run to avoid 'No module named src'
    # Example:
    #   python -m src.full_image_eval --cfg configs/config.yaml --run_dir ./runs/ijepa_coco2014_ooc_paste
    main(args.cfg, args.run_dir, auto_check_n=args.auto_check_n)
