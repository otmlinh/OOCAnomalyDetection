# I-JEPA COCO-OOC pipeline

## 0) Data layout (expected)
coco/
  train2017/
  val2017/
  annotations_trainval2017/
    instances_train2017.json
    instances_val2017.json
  coco_ooc/
    images/
    annotations/   # 106036 *.npy

## 1) Install
pip install -r requirements.txt

## 2) Continue pretrain (JEPA-like)
python -m src.train_ssl --cfg configs/config.yaml

Notes:
- Default config trains only last 2 blocks + predictor to avoid OOM.
- Outputs saved under runs/ijepa_coco_ooc/

## 3) Train supervised detector head
python -m src.train_detector --cfg configs/config.yaml

## 4) Evaluate
python -m src.eval_ooc --cfg configs/config.yaml

Outputs:
- log file with timestamps
- eval_scores.npz (y_true, y_score)
