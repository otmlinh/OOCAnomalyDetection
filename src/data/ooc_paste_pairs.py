#Dua tren ban goc.
# 3 loại bất thường OOC (Ngữ cảnh, Kích thước, Vị trí).
#Kỹ thuật làm mượt biên (Poisson Blending cho ngữ cảnh, Gaussian Blur Mask cho kích thước/vị trí).
#Data Augmentations thực thụ (Color Jittering cho vật thể và Overall Blurring cho toàn ảnh) để buộc mô hình không học vẹt các vết cắt ghép pixel.
#Luu 2 mau cho moi loai
#bat thuong vi tri: chia 3 vung Tri-level Inverse Positioning & Scaling)


import os
import random
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def safe_crop_rgb(pil_img: Image.Image, bbox, min_size: int = 16) -> Image.Image:
    """Trích xuất và resize vật thể từ ảnh gốc dựa trên bbox."""
    pil_img = pil_img.convert("RGB")
    x, y, w, h = bbox
    x0, y0 = int(max(0, x)), int(max(0, y))
    x1, y1 = int(min(pil_img.width, x + w)), int(min(pil_img.height, y + h))
    if x1 <= x0 or y1 <= y0:
        return pil_img
    crop = pil_img.crop((x0, y0, x1, y1)).convert("RGB")
    if crop.width < min_size or crop.height < min_size:
        crop = crop.resize((max(min_size, crop.width), max(min_size, crop.height)), resample=Image.BILINEAR)
    return crop

def clamp_bbox_xywh(x: int, y: int, w: int, h: int, W: int, H: int):
    """Đảm bảo tọa độ vật thể nằm trong phạm vi ảnh nền."""
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
    return [x, y, w, h]

class OOCPastePairsDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        image_ids: List[int],
        coco_instances,
        processor,
        pairs_per_image: int = 2,
        neg_ratio: int = 1,
        max_boxes_per_image: int = 10,
        min_crop_size: int = 16,
        paste_max_tries: int = 30,
        paste_jitter: float = 0.05,
        **kwargs # Đọc các tham số cấu hình động từ train_detector.py
    ):
        self.images_dir = images_dir
        self.image_ids = list(image_ids)
        self.coco = coco_instances
        self.processor = processor
        self.pairs_per_image = int(pairs_per_image)
        self.neg_ratio = int(neg_ratio)
        self.max_boxes_per_image = int(max_boxes_per_image)
        self.min_crop_size = int(min_crop_size)
        self.paste_max_tries = int(paste_max_tries)
        self.paste_jitter = float(paste_jitter)

        self.valid_imgs = [iid for iid in self.image_ids if self.coco.anns_for_image(iid)]
        if len(self.valid_imgs) < 2:
            raise ValueError("Cần ít nhất 2 ảnh có chú thích để thực hiện kỹ thuật Cut-and-Paste.")

        # --- ĐỌC CÁC THAM SỐ CẤU HÌNH TỪ KWARGS ---
        self.neg_variant = kwargs.get("neg_variant", "hybrid") 
        
        # Scale parameters
        self.scale_small_range = kwargs.get("scale_small_range", (0.2, 0.5))
        self.scale_large_range = kwargs.get("scale_large_range", (1.5, 3.0))
        self.scale_prob_small = float(kwargs.get("scale_prob_small", 0.1))
        
        # Hybrid probabilities [Context, Scale, Position]
        self.hybrid_probs = kwargs.get("hybrid_probs", [0.34, 0.33, 0.33])
        if self.hybrid_probs is None:
            self.hybrid_probs = [0.34, 0.33, 0.33]

        # Khởi tạo bộ Data Augmentation cho vật thể (Color Jittering)
        self.jitter_transform = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

        # --- [MỚI] BẢN ĐỒ NGỮ NGHĨA COCO 2017 CHO POSITION ANOMALY ---
        # Bầu trời/Trên cao: Máy bay(5), chim(16), đĩa ném(29), diều(38)
        self.SKY_IDS = {5, 16, 29, 38} 
        
        # Mặt đất/Dưới thấp: Người, xe cộ, động vật, nội thất lớn, v.v.
        self.GROUND_IDS = {
            1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
            52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 70, 72
        }
        # Các ID còn lại tự động rơi vào nhóm MIDDLE (Tầm trung)

    def __len__(self):
        per_img = self.pairs_per_image * (1 + self.neg_ratio)
        return len(self.valid_imgs) * per_img

    # --- CÁC HÀM XỬ LÝ BIÊN & DATA AUGMENTATION ---

    def _apply_augmentations(self, obj_img: Image.Image) -> Image.Image:
        if random.random() < 0.7:  
            obj_img = self.jitter_transform(obj_img)
        return obj_img

    def _apply_overall_blur(self, comp_img: Image.Image) -> Image.Image:
        if random.random() < 0.5:  
            radius = random.uniform(0.5, 1.5)
            comp_img = comp_img.filter(ImageFilter.GaussianBlur(radius=radius))
        return comp_img

    def _gaussian_blur_paste(self, bg_img: Image.Image, obj_crop: Image.Image, location: Tuple[int, int]) -> Image.Image:
        mask = Image.new("L", obj_crop.size, 255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
        comp = bg_img.copy()
        comp.paste(obj_crop, location, mask)
        return comp

    def _poisson_blending(self, bg_img: Image.Image, obj_crop: Image.Image, location: Tuple[int, int]) -> Image.Image:
        bg_cv = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2BGR)
        obj_cv = cv2.cvtColor(np.array(obj_crop), cv2.COLOR_RGB2BGR)
        mask = 255 * np.ones(obj_cv.shape, obj_cv.dtype)
        center = (location[0] + obj_cv.shape[1] // 2, location[1] + obj_cv.shape[0] // 2)
        try:
            mixed = cv2.seamlessClone(obj_cv, bg_cv, mask, center, cv2.NORMAL_CLONE)
            return Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
        except:
            return self._gaussian_blur_paste(bg_img, obj_crop, location)

    # --- CÁC CHIẾN LƯỢC SINH BẤT THƯỜNG ĐỘNG ---

    def _apply_context_anomaly(self, bg_img: Image.Image, obj_crop: Image.Image) -> Tuple[Image.Image, List[int]]:
        """Bất thường ngữ cảnh: Kích thước bình thường (scale 0.8-1.2), vị trí ngẫu nhiên."""
        W, H = bg_img.size
        scale = random.uniform(0.8, 1.2)
        target_w = max(self.min_crop_size, int(obj_crop.width * scale))
        target_h = max(self.min_crop_size, int(obj_crop.height * scale))
        
        obj_resized = obj_crop.resize((target_w, target_h), Image.BILINEAR)
        obj_resized = self._apply_augmentations(obj_resized)
        
        x = random.randint(0, max(0, W - target_w))
        y = random.randint(0, max(0, H - target_h))
        bbox = clamp_bbox_xywh(x, y, target_w, target_h, W, H)
        
        comp = self._poisson_blending(bg_img, obj_resized, (x, y))
        comp = self._apply_overall_blur(comp)
        
        self._save_debug_sample(comp, "context_anomaly")
        return comp, bbox

    def _apply_scale_anomaly(self, bg_img: Image.Image, obj_crop: Image.Image) -> Tuple[Image.Image, List[int]]:
        """Bất thường kích thước: Áp dụng scale_small_range và scale_large_range."""
        W, H = bg_img.size
        
        if random.random() < self.scale_prob_small:
            scale_factor = random.uniform(*self.scale_small_range)
        else:
            scale_factor = random.uniform(*self.scale_large_range)
            
        target_w = max(self.min_crop_size, int(obj_crop.width * scale_factor))
        target_w = min(target_w, W - 1)
        target_h = max(self.min_crop_size, int(obj_crop.height * (target_w / max(1, obj_crop.width))))
        
        obj_resized = obj_crop.resize((target_w, target_h), Image.BILINEAR)
        obj_resized = self._apply_augmentations(obj_resized)
        
        x = random.randint(0, max(0, W - target_w))
        y = random.randint(0, max(0, H - target_h))
        bbox = clamp_bbox_xywh(x, y, target_w, target_h, W, H)
        
        comp = self._gaussian_blur_paste(bg_img, obj_resized, (x, y))
        comp = self._apply_overall_blur(comp)
        
        self._save_debug_sample(comp, "scale_anomaly")
        return comp, bbox

    def _apply_position_anomaly(self, bg_img: Image.Image, obj_crop: Image.Image, category_id: int) -> Tuple[Image.Image, List[int]]:
        """Bất thường vị trí: Có nhận thức ngữ nghĩa (Tri-level Inverse Positioning & Scaling)."""
        W, H = bg_img.size
        
        # 1. Tri-level Mapping & Inverse Scaling
        if category_id in self.SKY_IDS:
            # Vật trên cao -> Ép dán lún xuống ĐẤT. Giữ nguyên kích thước.
            y_min, y_max = int(H * 0.7), H
            scale = random.uniform(0.8, 1.2)
            
        elif category_id in self.GROUND_IDS:
            # Vật mặt đất -> Ép dán bay lên TRỜI. 
            # Bắt buộc thu nhỏ để tạo cảm giác ở xa, ép mô hình bắt lỗi vị trí chứ ko bắt lỗi kích thước.
            y_min, y_max = 0, int(H * 0.2)
            scale = random.uniform(0.3, 0.6)
            
        else:
            # Vật tầm trung (cốc, sách, laptop) -> Ép dán lơ lửng trên tường/không trung.
            y_min, y_max = int(H * 0.2), int(H * 0.5)
            scale = random.uniform(0.8, 1.2)

        # 2. Xử lý kích thước mới
        target_w = max(self.min_crop_size, int(obj_crop.width * scale))
        target_h = max(self.min_crop_size, int(obj_crop.height * scale))
        
        obj_resized = obj_crop.resize((target_w, target_h), Image.BILINEAR)
        obj_resized = self._apply_augmentations(obj_resized)
        
        # 3. Tính toán tọa độ và dán
        x = random.randint(0, max(0, W - target_w))
        y = random.randint(y_min, max(y_min, y_max - target_h))
        bbox = clamp_bbox_xywh(x, y, target_w, target_h, W, H)
        
        comp = self._gaussian_blur_paste(bg_img, obj_resized, (x, y))
        comp = self._apply_overall_blur(comp)
        
        self._save_debug_sample(comp, "position_anomaly")
        return comp, bbox

    # --- HÀM DEBUG VÀ HELPER ---

    def _save_debug_sample(self, composite_img: Image.Image, anomaly_name: str):
        debug_dir = "debug_samples"
        os.makedirs(debug_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(debug_dir) if f.startswith(anomaly_name)]
        if len(existing_files) < 2:
            sample_idx = len(existing_files) + 1
            save_path = os.path.join(debug_dir, f"{anomaly_name}_sample_{sample_idx}.png")
            composite_img.save(save_path)

    def _load_rgb(self, image_id: int) -> Image.Image:
        fn = self.coco.image_file(image_id)
        path = os.path.join(self.images_dir, fn)
        return Image.open(path).convert("RGB")

    def _pick_ann(self, image_id: int) -> Dict[str, Any]:
        anns = self.coco.anns_for_image(image_id)
        if len(anns) > self.max_boxes_per_image:
            anns = random.sample(anns, self.max_boxes_per_image)
        return random.choice(anns)

    def _process_pair(self, obj_img, ctx_img, label):
        obj = self.processor(images=obj_img, return_tensors="pt")
        ctx = self.processor(images=ctx_img, return_tensors="pt")
        return {
            "obj_pixel_values": obj["pixel_values"].squeeze(0),
            "ctx_pixel_values": ctx["pixel_values"].squeeze(0),
            "label": label
        }

    def __getitem__(self, idx):
        per_img = self.pairs_per_image * (1 + self.neg_ratio)
        img_idx = idx // per_img
        rem = idx % per_img
        is_neg = rem >= self.pairs_per_image

        target_id = self.valid_imgs[img_idx]
        target_img = self._load_rgb(target_id)

        if not is_neg:
            ann_t = self._pick_ann(target_id)
            obj_img = safe_crop_rgb(target_img, ann_t["bbox"], min_size=self.min_crop_size)
            return self._process_pair(obj_img, target_img, 0)

        # Quyết định biến thể OOC sẽ sử dụng cho ảnh này
        variant = self.neg_variant
        if variant == "hybrid":
            variant = random.choices(
                ["context", "scale", "position"], 
                weights=self.hybrid_probs, 
                k=1
            )[0]

        for _ in range(self.paste_max_tries):
            src_id = random.choice(self.valid_imgs)
            if src_id == target_id: continue

            src_img = self._load_rgb(src_id)
            ann_s = self._pick_ann(src_id)
            obj_crop = safe_crop_rgb(src_img, ann_s["bbox"], min_size=self.min_crop_size)

            # Áp dụng hàm sinh OOC tương ứng
            if variant in ["context", "paste"]:
                comp, pasted_bbox = self._apply_context_anomaly(target_img, obj_crop)
            elif variant in ["scale", "paste_scale"]:
                comp, pasted_bbox = self._apply_scale_anomaly(target_img, obj_crop)
            elif variant in ["position", "paste_misplace"]:
                # --- TRUYỀN CATEGORY ID VÀO HÀM ---
                comp, pasted_bbox = self._apply_position_anomaly(target_img, obj_crop, ann_s["category_id"])
            else:
                comp, pasted_bbox = self._apply_context_anomaly(target_img, obj_crop) # Fallback

            obj_img = safe_crop_rgb(comp, pasted_bbox, min_size=self.min_crop_size)
            return self._process_pair(obj_img, comp, 1)

        return self._process_pair(target_img, target_img, 0)