from netaug.netaug import NetAugTrainer

overrides = dict(data="yolov8_lighter/bdd100k.yaml",
                 model="netaug/yolov8n_dynamic.yaml",
                 imgsz=640,
                 epochs=1200,
                 batch=64,
                 save=True,
                 exist_ok=True,
                 project='runs',
                 name='net_aug/yolov8w015_dynamic_5x',
                 device=0)

trainer = NetAugTrainer(overrides=overrides, cfg="netaug/default.yaml")
trainer.train()
