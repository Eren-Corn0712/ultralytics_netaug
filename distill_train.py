from distllation.distill import DistillationTrainer

overrides = dict(data="yolov8_lighter/bdd100k.yaml",
                 model="yolov8_lighter/yolov8_lighter.yaml",
                 imgsz=640,
                 epochs=1200,
                 batch=64,
                 save=True,
                 exist_ok=True,
                 project='runs/yolov8_logit_kd',
                 name='kd_feat_s',
                 device=1)

trainer = DistillationTrainer(overrides=overrides, cfg="yolov8_lighter/default.yaml")
trainer.train()
