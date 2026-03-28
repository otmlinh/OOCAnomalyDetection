import os, glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CocoOOCDataset(Dataset):
    """
    Loads OOC composite images + .npy annotations.
    Each npy is a 0-d object array containing dict:
      {'original_ann_ids': [...], 'image_id': <bg_image_id>, 'ooc_annotation': {'bbox':[x,y,w,h], ...}}
    """
    def __init__(self, ooc_images_dir, ooc_ann_dir, processor, max_items=-1):
        self.ooc_images_dir = ooc_images_dir
        self.ooc_ann_dir = ooc_ann_dir
        self.processor = processor

        self.npy_paths = sorted(glob.glob(os.path.join(ooc_ann_dir, "*.npy")))
        if max_items is not None and max_items > 0:
            self.npy_paths = self.npy_paths[:max_items]

    def __len__(self):
        return len(self.npy_paths)

    def _find_image_path(self, stem: str):
        # common extensions
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(self.ooc_images_dir, stem + ext)
            if os.path.exists(p):
                return p
        # fallback: glob by prefix
        cand = glob.glob(os.path.join(self.ooc_images_dir, stem + ".*"))
        if len(cand) > 0:
            return cand[0]
        raise FileNotFoundError(f"Cannot find image for stem={stem} in {self.ooc_images_dir}")

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        name = os.path.basename(npy_path)
        stem = os.path.splitext(name)[0]

        arr = np.load(npy_path, allow_pickle=True)
        entry = arr.item() if isinstance(arr, np.ndarray) and arr.ndim == 0 else arr

        img_path = self._find_image_path(stem)
        img = Image.open(img_path).convert("RGB")
        enc = self.processor(images=img, return_tensors="pt")

        return {
            "image_path": img_path,
            "pixel_values": enc["pixel_values"].squeeze(0),
            "entry": entry,
        }
