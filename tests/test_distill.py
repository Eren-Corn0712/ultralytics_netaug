from distllation.distill import DistillationTrainer
from ultralytics.yolo.utils import (LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save)

CFG = ROOT.parent / 'yolov8_lighter' / 'yolov8_lighter.yaml'
DATA = ROOT.parent / 'yolov8_lighter' / 'bdd100k.yaml'
DEFAULT_CFG = ROOT.parent / 'yolov8_lighter' / 'default.yaml'
TEACHER_WEIGHT = ROOT.parent / 'tests/runs/teacher/weights/best.pt'


class TestDistillation(object):
    def __init__(self, *args, **kwargs):
        pass

    def test_distillation_trainer(self, *args, **kwargs):
        overrides = dict(data=str(DATA),
                         model=CFG,
                         imgsz=640,
                         epochs=1,
                         batch=16,
                         save=False,
                         exist_ok=True,
                         project='runs',
                         name='debug',
                         teacher_weights=TEACHER_WEIGHT,
                         device="0")

        trainer = DistillationTrainer(overrides=overrides, cfg=DEFAULT_CFG)
        trainer.train()

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


if __name__ == "__main__":
    LOGGER.info("Start Test the distillation!")
    test_class = TestDistillation()
    test_class()
    LOGGER.info("No Expected exception caught")
    LOGGER.info("All the test passed!")
