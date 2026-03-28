import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import load_config, ensure_dir
from src.utils.seed import seed_all
from src.utils.logger import Logger
from src.utils.meter import AverageMeter

from src.data.coco_instances import CocoInstances
from src.data.coco_images import CocoImageDataset
from src.models.ijepa_backbone import IJepaBackbone
from src.models.jepa_ssl import JEPAContinuedPretrain


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


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_n_blocks(model, n: int):
    freeze_all(model)
    layers = _get_encoder_layers(model)
    if layers is None or n <= 0:
        return 0
    n = min(n, len(layers))
    for blk in layers[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(cfg_path="configs/config.yaml"):
    cfg = load_config(cfg_path)
    seed_all(cfg["project"]["seed"])

    out_root = cfg["project"]["out_dir"]
    ensure_dir(out_root)
    run_dir = os.path.join(out_root, cfg["project"]["name"])
    ensure_dir(run_dir)
    logger = Logger(run_dir, "train_ssl")

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["runtime"]["amp"])
    amp_dtype = torch.bfloat16 if cfg["runtime"]["amp_dtype"].lower() == "bf16" else torch.float16

    coco_train = CocoInstances(cfg["paths"]["instances_train"])
    train_ids = list(coco_train.img_by_id.keys())

    # IMPORTANT: use_mask_token=True to support bool_masked_pos :contentReference[oaicite:3]{index=3}
    backbone = IJepaBackbone(
        cfg["hf"]["model_id"],
        attn_implementation=cfg["hf"].get("attn_implementation", "sdpa"),
        gradient_checkpointing=bool(cfg["ssl"].get("gradient_checkpointing", False)),
        use_mask_token=True,
    )
    backbone.to(device)

    # Unfreeze last N blocks for actual SSL adaptation (avoid "no requires_grad" issues)
    n_train = int(cfg["ssl"]["train_last_n_blocks"])
    trainable_bb = unfreeze_last_n_blocks(backbone.model, n_train)
    backbone.model.train()
    logger.log(f"Unfrozen last {n_train} block(s). Trainable backbone params: {trainable_bb/1e6:.2f}M")

    # Infer embed dim
    with torch.no_grad():
        dummy = torch.zeros((1, 3, cfg["image"]["size"], cfg["image"]["size"]), device=device)
        tokens = backbone.forward_tokens(dummy)  # (1,T,D)
        dim = tokens.shape[-1]

    ssl = JEPAContinuedPretrain(backbone, embed_dim=dim, ema_momentum=float(cfg["ssl"]["ema_momentum"]))
    ssl.to(device)
    ssl.train()

    params = [p for p in ssl.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["ssl"]["lr"]), weight_decay=float(cfg["ssl"]["weight_decay"]))

    ds = CocoImageDataset(cfg["paths"]["train_images"], train_ids, coco_train, transform=None)

    def collate(batch):
        imgs = [b["image"] for b in batch]
        enc = backbone.processor(imgs, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"]}

    dl = DataLoader(
        ds,
        batch_size=int(cfg["ssl"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=bool(cfg["runtime"]["pin_memory"]),
        collate_fn=collate,
        drop_last=True,
    )

    logger.log(f"Device={device}, AMP={use_amp}/{cfg['runtime']['amp_dtype']}")
    logger.log(f"SSL epochs={cfg['ssl']['epochs']} bs={cfg['ssl']['batch_size']} grad_accum={cfg['ssl']['grad_accum']}")
    logger.log(f"Optim params: {sum(p.numel() for p in params)/1e6:.1f}M")

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    step = 0
    loss_meter = AverageMeter()

    for ep in range(int(cfg["ssl"]["epochs"])):
        pbar = tqdm(dl, desc=f"ssl ep{ep}", dynamic_ncols=True)
        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = ssl(pixel_values, mask_ratio=float(cfg["ssl"]["mask_ratio"]))

            loss_scaled = loss / int(cfg["ssl"]["grad_accum"])

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (it + 1) % int(cfg["ssl"]["grad_accum"]) == 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            loss_meter.update(loss.item(), k=pixel_values.size(0))
            step += 1
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

            if step % int(cfg["ssl"]["save_every_steps"]) == 0:
                ckpt = os.path.join(run_dir, f"ssl_step{step}.pt")
                torch.save({"student": ssl.student.model.state_dict(),
                            "predictor": ssl.predictor.state_dict(),
                            "cfg": cfg}, ckpt)
                logger.log(f"Saved {ckpt}")

        logger.log(f"Epoch {ep} done. avg_loss={loss_meter.avg:.4f}")

    ckpt = os.path.join(run_dir, "ssl_final.pt")
    torch.save({"student": ssl.student.model.state_dict(),
                "predictor": ssl.predictor.state_dict(),
                "cfg": cfg}, ckpt)
    logger.log(f"Saved {ckpt}")
    logger.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.cfg)
