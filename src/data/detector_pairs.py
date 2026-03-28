import os
import random
from PIL import Image
from torch.utils.data import Dataset

def crop_with_bbox(pil_img: Image.Image, bbox, min_size: int = 16):
    """
    bbox: [x,y,w,h] in COCO format
    Ensure:
      - valid crop region
      - RGB output
      - minimum crop size (to avoid tiny/ambiguous channel dimension issues)
    """
    pil_img = pil_img.convert("RGB")

    x, y, w, h = bbox
    x0 = int(max(0, x))
    y0 = int(max(0, y))
    x1 = int(min(pil_img.width, x + w))
    y1 = int(min(pil_img.height, y + h))

    # invalid bbox -> fallback full image
    if x1 <= x0 or y1 <= y0:
        return pil_img

    crop = pil_img.crop((x0, y0, x1, y1)).convert("RGB")

    # If crop too small -> resize up (or fallback)
    if crop.width < min_size or crop.height < min_size:
        # resize up to min_size keeping aspect ratio-ish (simple)
        new_w = max(min_size, crop.width)
        new_h = max(min_size, crop.height)
        crop = crop.resize((new_w, new_h), resample=Image.BILINEAR).convert("RGB")

    return crop

class DetectorPairsDataset(Dataset):
    """
    Supervised head training without synthesizing OOC images.
    Negative: (object crop, its own context image) label=0 (in-context)
    Positive : (object crop, random other context image) label=1 (OOC-like)
    """
    def __init__(self, images_dir, image_ids, coco_instances, processor,
                 pairs_per_image=2, neg_ratio=1, max_boxes_per_image=10,
                 min_crop_size=16, max_resample_tries=10):
        self.images_dir = images_dir
        self.image_ids = list(image_ids)
        self.coco = coco_instances
        self.processor = processor
        self.pairs_per_image = pairs_per_image
        self.neg_ratio = neg_ratio
        self.max_boxes_per_image = max_boxes_per_image
        self.min_crop_size = int(min_crop_size)
        self.max_resample_tries = int(max_resample_tries)

        self.valid_imgs = []
        for iid in self.image_ids:
            anns = self.coco.anns_for_image(iid)
            if len(anns) > 0:
                self.valid_imgs.append(iid)

    def __len__(self):
        per_img = self.pairs_per_image * (1 + self.neg_ratio)
        return len(self.valid_imgs) * per_img

    def _load_rgb(self, image_id: int) -> Image.Image:
        fn = self.coco.image_file(image_id)
        path = os.path.join(self.images_dir, fn)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        per_img = self.pairs_per_image * (1 + self.neg_ratio)
        img_idx = idx // per_img
        rem = idx % per_img
        is_neg = rem >= self.pairs_per_image
        base_id = self.valid_imgs[img_idx]

        # We will try a few times in case bbox is degenerate / processor fails
        for _ in range(self.max_resample_tries):
            base_img = self._load_rgb(base_id)

            anns = self.coco.anns_for_image(base_id)
            if len(anns) > self.max_boxes_per_image:
                anns = random.sample(anns, self.max_boxes_per_image)
            ann = random.choice(anns)

            obj_img = crop_with_bbox(base_img, ann["bbox"], min_size=self.min_crop_size)

            if not is_neg:
                ctx_img = base_img
                label = 0
            else:
                other_id = random.choice(self.valid_imgs)
                while other_id == base_id:
                    other_id = random.choice(self.valid_imgs)
                ctx_img = self._load_rgb(other_id)
                label = 1

            # Extra safety: enforce RGB (again)
            obj_img = obj_img.convert("RGB")
            ctx_img = ctx_img.convert("RGB")

            try:
                obj = self.processor(images=obj_img, return_tensors="pt")
                ctx = self.processor(images=ctx_img, return_tensors="pt")
                return {
                    "obj_pixel_values": obj["pixel_values"].squeeze(0),
                    "ctx_pixel_values": ctx["pixel_values"].squeeze(0),
                    "label": label,
                }
            except Exception:
                # try another bbox/image pairing
                continue

        # If still failing, fallback to full image pairing (very rare)
        base_img = self._load_rgb(base_id)
        obj = self.processor(images=base_img, return_tensors="pt")
        ctx = self.processor(images=base_img, return_tensors="pt")
        return {
            "obj_pixel_values": obj["pixel_values"].squeeze(0),
            "ctx_pixel_values": ctx["pixel_values"].squeeze(0),
            "label": 0,
        }
