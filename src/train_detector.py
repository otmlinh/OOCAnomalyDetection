#Cross-Attention, 
#chia tập dữ liệu thành Train/Validation (tỷ lệ 9:1), 
#evaluate_model: tính toán các chỉ số (Loss, Accuracy, Precision, Recall, F1-Score), và logic 
#Lưu mô hình tốt nhất (Best Model) dựa trên AUROC.

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
# [ĐÃ THÊM] roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.io import load_config, ensure_dir
from src.utils.seed import seed_all
from src.utils.logger import Logger
from src.utils.meter import AverageMeter

from src.data.coco_instances import CocoInstances

# BOTH datasets supported:
from src.data.detector_pairs import DetectorPairsDataset         
from src.data.ooc_paste_pairs import OOCPastePairsDataset        

from src.models.ijepa_backbone import IJepaBackbone
from src.models.detector_head import OOCDetectorHead


def _get_encoder_layers(hf_model):
    if hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layer"):
        return hf_model.encoder.layer
    if hasattr(hf_model, "vision_model"):
        vm = hf_model.vision_model
        if hasattr(vm, "encoder"):
            enc = vm.encoder
            if hasattr(enc, "layer"):
                return enc.layer
            if hasattr(enc, "layers"):
                return enc.layers
    if hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layers"):
        return hf_model.encoder.layers
    return None

def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_n_blocks(model: torch.nn.Module, n: int) -> int:
    freeze_all(model)
    layers = _get_encoder_layers(model)
    if layers is None or n <= 0:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    n = min(n, len(layers))
    for blk in layers[-n:]:
        for p in blk.parameters():
            p.requires_grad = True

    for name, module in model.named_modules():
        if name.lower().endswith("norm") or "final" in name.lower():
            for p in module.parameters():
                p.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_model(head, backbone, dataloader, device, amp_dtype, use_amp, freeze_backbone):
    head.eval()
    if not freeze_backbone:
        backbone.model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] # [ĐÃ THÊM] Lưu xác suất để tính AUROC

    for batch in dataloader:
        obj = batch["obj"].to(device, non_blocking=True)
        ctx = batch["ctx"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            obj_emb = backbone.encode_object(obj)         
            ctx_tokens = backbone.encode_context(ctx)     
            
            logits = head(obj_emb, ctx_tokens)
            loss = F.binary_cross_entropy_with_logits(logits, y)

        total_loss += loss.item() * obj.size(0)
        
        # Tính xác suất (prob) và dự đoán nhãn (pred)
        probs = torch.sigmoid(logits).float()
        preds = (logits > 0).float()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    head.train()
    if not freeze_backbone:
        backbone.model.train()

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # [ĐÃ THÊM] Tính AUROC
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0 # Phòng hờ trường hợp batch val chỉ có 1 class (rất hiếm)

    return avg_loss, acc, prec, rec, f1, auroc


def main(cfg_path="configs/config.yaml", resume=False):
    cfg = load_config(cfg_path)
    seed_all(cfg["project"]["seed"])

    out_root = cfg["project"]["out_dir"]
    ensure_dir(out_root)
    run_dir = os.path.join(out_root, cfg["project"]["name"])
    ensure_dir(run_dir)
    logger = Logger(run_dir, "train_detector")

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["runtime"]["amp"])
    amp_dtype = torch.bfloat16 if cfg["runtime"]["amp_dtype"].lower() == "bf16" else torch.float16

    # 1. LOAD DATA & SPLIT TRAIN/VAL
    coco_train = CocoInstances(cfg["paths"]["instances_train"])
    all_train_ids = list(coco_train.img_by_id.keys())
    
    train_ids, val_ids = train_test_split(all_train_ids, test_size=0.1, random_state=cfg["project"]["seed"])
    logger.log(f"Data Split: {len(train_ids)} Train images | {len(val_ids)} Validation images")

    # 2. INIT BACKBONE
    backbone = IJepaBackbone(
        cfg["hf"]["model_id"],
        attn_implementation=cfg["hf"].get("attn_implementation", "sdpa"),
        gradient_checkpointing=bool(cfg["detector"].get("gradient_checkpointing", False)),
    )
    backbone.to(device)

    ckpt_dir = os.path.abspath(os.path.expanduser(cfg["project"].get("ckpt_dir", run_dir)))
    ssl_ckpt = os.path.join(ckpt_dir, "ssl_final.pt")
    if os.path.exists(ssl_ckpt):
        sd = torch.load(ssl_ckpt, map_location="cpu")
        missing, unexpected = backbone.model.load_state_dict(sd["student"], strict=False)
        logger.log(f"Loaded SSL backbone from {ssl_ckpt}. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        logger.log("No SSL checkpoint found. Using pure HF pretrained backbone.")

    freeze_backbone = bool(cfg["detector"]["freeze_backbone"])
    unfreeze_n = int(cfg["detector"].get("unfreeze_last_n_blocks", 0))

    if freeze_backbone:
        freeze_all(backbone.model)
        backbone.model.eval()
        logger.log("Backbone is frozen (eval mode).")
    else:
        trainable_params = unfreeze_last_n_blocks(backbone.model, unfreeze_n)
        backbone.model.train()
        logger.log(
            f"Backbone finetune enabled. Unfrozen last {unfreeze_n} block(s). "
            f"Trainable params: {trainable_params/1e6:.2f}M"
        )

    with torch.no_grad():
        dummy = torch.zeros((1, 3, cfg["image"]["size"], cfg["image"]["size"]), device=device)
        z = backbone.encode_object(dummy)
        dim = z.shape[-1]

    # 3. INIT DETECTOR HEAD
    head = OOCDetectorHead(
        dim=dim,
        num_heads=int(cfg["detector"].get("num_heads", 8)),
        hidden=int(cfg["detector"]["head_hidden"]),
        dropout=float(cfg["detector"]["dropout"])
    )
    head.to(device)
    head.train()

    # 4. SETUP OPTIMIZER
    head_lr = float(cfg["detector"]["head_lr"])
    bb_lr = float(cfg["detector"]["backbone_lr"])
    wd = float(cfg["detector"]["weight_decay"])

    param_groups = [{"params": head.parameters(), "lr": head_lr, "weight_decay": wd}]
    if not freeze_backbone:
        bb_params = [p for p in backbone.model.parameters() if p.requires_grad]
        param_groups.append({"params": bb_params, "lr": bb_lr, "weight_decay": wd})

    opt = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # --- LOGIC RESUME TẬP TRUNG VÀO AUROC ---
    start_epoch = 0
    best_val_auroc = 0.0 # [ĐÃ ĐỔI]

    if resume:
        resume_ckpt = os.path.join(run_dir, "detector_last.pt")
        if os.path.exists(resume_ckpt):
            logger.log(f"Đang nạp trạng thái từ file {resume_ckpt} để tiếp tục train...")
            checkpoint = torch.load(resume_ckpt, map_location=device)
            
            head.load_state_dict(checkpoint["head"])
            if not freeze_backbone and checkpoint["backbone"] is not None:
                backbone.model.load_state_dict(checkpoint["backbone"])
            
            if "optimizer" in checkpoint:
                opt.load_state_dict(checkpoint["optimizer"])
            if use_amp and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
                
            start_epoch = checkpoint["epoch"] + 1
            if "best_val_auroc" in checkpoint:
                best_val_auroc = checkpoint["best_val_auroc"] # [ĐÃ ĐỔI]
                
            logger.log(f"Tiếp tục thành công! Sẽ bắt đầu train từ Epoch {start_epoch}. Best AUROC hiện tại: {best_val_auroc:.4f}")
        else:
            logger.log(f"Cảnh báo: Không tìm thấy {resume_ckpt}. Sẽ bắt đầu train lại từ đầu (Epoch 0).")

    # 5. SETUP DATASETS & DATALOADERS
    neg_mode = str(cfg["detector"].get("neg_mode", "swap")).lower()
    min_crop = int(cfg["image"].get("min_crop_size", 16))
    
    if neg_mode == "paste":
        DatasetClass = OOCPastePairsDataset
        ds_kwargs = {
            "images_dir": cfg["paths"]["train_images"],
            "coco_instances": coco_train,
            "processor": backbone.processor,
            "pairs_per_image": int(cfg["detector"]["pairs_per_image"]),
            "neg_ratio": int(cfg["detector"]["neg_ratio"]),
            "max_boxes_per_image": int(cfg["image"]["max_boxes_per_image"]),
            "min_crop_size": min_crop,
            "paste_max_tries": int(cfg["detector"].get("paste_max_tries", 30)),
            "paste_jitter": float(cfg["detector"].get("paste_jitter", 0.05)),
        }
        
        for k in ["neg_variant", "scale_small_range", "scale_large_range", "scale_prob_small", 
                  "misplace_band", "misplace_prob_top", "hybrid_probs"]:
            if k in cfg["detector"]:
                ds_kwargs[k] = cfg["detector"][k]
    else:
        DatasetClass = DetectorPairsDataset
        ds_kwargs = {
            "images_dir": cfg["paths"]["train_images"],
            "coco_instances": coco_train,
            "processor": backbone.processor,
            "pairs_per_image": int(cfg["detector"]["pairs_per_image"]),
            "neg_ratio": int(cfg["detector"]["neg_ratio"]),
            "max_boxes_per_image": int(cfg["image"]["max_boxes_per_image"]),
            "min_crop_size": min_crop,
        }

    train_ds = DatasetClass(image_ids=train_ids, **ds_kwargs)
    
    val_kwargs = ds_kwargs.copy()
    val_kwargs["pairs_per_image"] = max(1, ds_kwargs["pairs_per_image"] // 2) 
    val_ds = DatasetClass(image_ids=val_ids, **val_kwargs)

    def collate(batch):
        obj = torch.stack([b["obj_pixel_values"] for b in batch], dim=0)
        ctx = torch.stack([b["ctx_pixel_values"] for b in batch], dim=0)
        y = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        return {"obj": obj, "ctx": ctx, "y": y}

    train_dl = DataLoader(train_ds, batch_size=int(cfg["detector"]["batch_size"]), shuffle=True,
                          num_workers=int(cfg["runtime"]["num_workers"]), pin_memory=bool(cfg["runtime"]["pin_memory"]),
                          collate_fn=collate, drop_last=True)

    val_dl = DataLoader(val_ds, batch_size=int(cfg["detector"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["runtime"]["num_workers"]), pin_memory=bool(cfg["runtime"]["pin_memory"]),
                        collate_fn=collate, drop_last=False)

    loss_meter = AverageMeter()
    grad_accum = int(cfg["detector"]["grad_accum"])
    clip_gn = float(cfg["detector"].get("clip_grad_norm", 0.0))

    opt.zero_grad(set_to_none=True)
    global_step = 0
    total_epochs = int(cfg["detector"]["epochs"])

    # 6. TRAINING LOOP
    for ep in range(start_epoch, total_epochs):
        pbar = tqdm(train_dl, desc=f"Train Ep{ep}/{total_epochs-1}", dynamic_ncols=True)
        loss_meter.reset()

        # --- TRAIN ---
        for it, batch in enumerate(pbar):
            obj = batch["obj"].to(device, non_blocking=True)
            ctx = batch["ctx"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                if freeze_backbone:
                    with torch.no_grad():
                        obj_emb = backbone.encode_object(obj)         
                        ctx_tokens = backbone.encode_context(ctx)     
                else:
                    obj_emb = backbone.encode_object(obj)             
                    ctx_tokens = backbone.encode_context(ctx)         
                
                logits = head(obj_emb, ctx_tokens)
                loss = F.binary_cross_entropy_with_logits(logits, y)

            loss_scaled = loss / grad_accum

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (it + 1) % grad_accum == 0:
                if clip_gn and clip_gn > 0:
                    if use_amp and amp_dtype == torch.float16:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=clip_gn)
                    if not freeze_backbone:
                        bb_params = [p for p in backbone.model.parameters() if p.requires_grad]
                        if len(bb_params) > 0:
                            torch.nn.utils.clip_grad_norm_(bb_params, max_norm=clip_gn)

                if use_amp and amp_dtype == torch.float16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)
                global_step += 1

            loss_meter.update(loss.item(), k=obj.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        logger.log(f"Epoch {ep} Train | Avg Loss: {loss_meter.avg:.4f}")

        # --- VALIDATION ---
        logger.log(f"Epoch {ep} Val   | Đang đánh giá...")
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auroc = evaluate_model(
            head, backbone, val_dl, device, amp_dtype, use_amp, freeze_backbone
        )
        
        # In thêm chỉ số AUROC ra Terminal và file log
        logger.log(
            f"Epoch {ep} Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
            f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | AUROC: {val_auroc:.4f}"
        )

        # [ĐÃ ĐỔI] Dùng AUROC làm tiêu chí lưu mô hình
        is_best = val_auroc > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc

        checkpoint_dict = {
            "epoch": ep,
            "head": head.state_dict(),
            "backbone": backbone.model.state_dict() if not freeze_backbone else None,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "best_val_auroc": best_val_auroc, # [ĐÃ ĐỔI]
            "cfg": cfg,
        }

        # Lưu best model (theo AUROC)
        if is_best:
            ckpt_best = os.path.join(run_dir, "detector_best.pt")
            best_dict = checkpoint_dict.copy()
            best_dict["metrics"] = {"val_loss": val_loss, "val_f1": val_f1, "val_acc": val_acc, "val_auroc": val_auroc}
            torch.save(best_dict, ckpt_best)
            logger.log(f"--> Đã lưu Best Model mới (AUROC: {val_auroc:.4f}) tại {ckpt_best}")

        # Lưu last model (dùng để resume)
        ckpt_last = os.path.join(run_dir, "detector_last.pt")
        torch.save(checkpoint_dict, ckpt_last)

    logger.log(f"Huấn luyện hoàn tất! Best AUROC đạt được: {best_val_auroc:.4f}")
    logger.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--resume", action="store_true", help="Tiếp tục train từ file detector_last.pt")
    args = ap.parse_args()
    
    main(cfg_path=args.cfg, resume=args.resume)