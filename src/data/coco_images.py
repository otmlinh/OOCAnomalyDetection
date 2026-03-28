import os
from PIL import Image
from torch.utils.data import Dataset

class CocoImageDataset(Dataset):
    def __init__(self, images_dir: str, image_ids, coco_instances, transform=None):
        self.images_dir = images_dir
        self.image_ids = list(image_ids)
        self.coco = coco_instances
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        fn = self.coco.image_file(image_id)
        path = os.path.join(self.images_dir, fn)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image_id": image_id, "image": img}
