from ultralytics import YOLO
from pathlib import Path

m_list = ['yolov8w015.yaml', 'yolov8n.pt']

for m in m_list[::-1]:
    model = YOLO(m)
    model.info(verbose=True)
    overrides = dict(
        data='yolov8_lighter/bdd100k.yaml',
        epochs=1200,
        batch=64,
        cache=False,
        device=1,
        workers=16,
        project='runs/yolov8',
        name=str(Path(m).with_suffix('')),
        exist_ok=True,
        lr0=0.01,
        lrf=0.001,
    )
    model.train(**overrides)
