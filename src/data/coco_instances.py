import json
from collections import defaultdict

class CocoInstances:
    """
    Load COCO instances json and build:
    - img_id -> list of annotations
    - ann_id -> annotation
    - img_id -> file_name
    """
    def __init__(self, instances_json_path: str):
        with open(instances_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.ann_by_id = {a["id"]: a for a in coco["annotations"]}
        self.img_by_id = {im["id"]: im for im in coco["images"]}
        self.cat_by_id = {c["id"]: c for c in coco["categories"]}

        self.anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            self.anns_by_img[a["image_id"]].append(a)

    def image_file(self, image_id: int) -> str:
        return self.img_by_id[image_id]["file_name"]

    def anns_for_image(self, image_id: int):
        return self.anns_by_img.get(image_id, [])

    def ann(self, ann_id: int):
        return self.ann_by_id.get(ann_id, None)
