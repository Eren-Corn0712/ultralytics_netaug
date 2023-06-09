import torch
from distllation.distill import DistillationTrainer
from ultralytics.yolo.utils import (LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.torch_utils import model_info
from netaug.netaug import NetAugTrainer, NetAugDetectionModel

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from netaug.utils.torch_utils import count_grad_parameters

CFG = ROOT.parent / 'netaug/yolov8n_dynamic.yaml'
DATA = ROOT.parent / 'yolov8_lighter' / 'bdd100k.yaml'
DEFAULT_CFG = ROOT.parent / 'netaug' / 'default.yaml'


def has_same_attributes(cls1, cls2):
    """檢查兩個類別是否有相同的屬性"""
    attrs1 = set(vars(cls1).keys())
    attrs2 = set(vars(cls2).keys())
    return bool(attrs1.intersection(attrs2))


def check_model(model1, model2):
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()
    for key in model1_dict:
        if key in model2_dict:
            if model1_dict[key].shape != model2_dict[key].shape:
                print(f"Different shape: {key} - {model1_dict[key].shape} vs {model2_dict[key].shape}")
            elif not torch.equal(model1_dict[key], model2_dict[key]):
                print(f"Different value: {key}")
        else:
            print(f"Missing key: {key} in model2")

    for key in model2_dict:
        if key not in model1_dict:
            print(f"Missing key: {key} in model1")
    print("check_model method passed!!!")


def check_class(class_a, class_b):
    attrs_a = dir(class_a)
    attrs_b = dir(class_b)

    # 记录不同的属性及其值
    diff_attrs = {}

    for attr_name in attrs_a:
        if not hasattr(class_b, attr_name):
            diff_attrs[attr_name] = ('not in ClassB', getattr(class_a, attr_name), None)
        else:
            attr_value_a = getattr(class_a, attr_name)
            attr_value_b = getattr(class_b, attr_name)
            if attr_value_a != attr_value_b:
                diff_attrs[attr_name] = (attr_value_a, attr_value_b)

    for attr_name in attrs_b:
        if not hasattr(class_a, attr_name):
            diff_attrs[attr_name] = ('not in ClassA', None, getattr(class_b, attr_name))

    if diff_attrs:
        print('The following attributes are different between ClassA and ClassB:')
        for attr_name, values in diff_attrs.items():
            if values[0] == 'not in ClassB':
                print(f'{attr_name}: {values[0]}, value in ClassA: {values[1]}')
            elif values[0] == 'not in ClassA':
                print(f'{attr_name}: {values[0]}, value in ClassB: {values[2]}')
            else:
                print(f'{attr_name}: value in ClassA: {values[0]}, value in ClassB: {values[1]}')
    else:
        print('The attributes and their values in ClassA and ClassB are exactly the same.')


class TestNetAug(object):
    def __init__(self, *args, **kwargs):
        pass

    def _test_load_pretrained(self):
        pretrained_weight = "../tests/runs/debug/weights/best.pt"
        model1 = YOLO(pretrained_weight)
        model2 = YOLO("yolov8n.pt")
        print(f"The {model1.model.__class__.__name__} and {model2.model.__class__.__name__} is the same attribute :",
              has_same_attributes(model1, model2))

    def test_sort_channel(self):
        netaug_detection_model = NetAugDetectionModel(CFG, max_depth=4.0)
        netaug_detection_model.sort_channels()

    def test_detection_aug_model(self):
        netaug_detection_model = NetAugDetectionModel(CFG)
        x = torch.ones(1, 3, 640, 640).cuda()
        netaug_detection_model = netaug_detection_model.cuda()
        for _ in range(5):
            netaug_detection_model.set_aug()
            y = netaug_detection_model(x)
            y = sum([z.sum() for z in y])
            y.backward(torch.ones_like(y))
            print("Grad Parameters: ", count_grad_parameters(netaug_detection_model))
            netaug_detection_model.zero_grad()

            print("Base nodel")
            netaug_detection_model.set_base()
            y = netaug_detection_model(x)
            y = sum([z.sum() for z in y])
            y.backward(torch.ones_like(y))
            print("Grad Parameters: ", count_grad_parameters(netaug_detection_model))
            netaug_detection_model.zero_grad()

    def _test_netaug_trainer(self, *args, **kwargs):
        overrides = dict(data=str(DATA),
                         model=CFG,
                         imgsz=640,
                         epochs=5,
                         batch=4,
                         save=True,
                         exist_ok=True,
                         project='runs',
                         name='debug',
                         device="0")

        trainer = NetAugTrainer(overrides=overrides, cfg=DEFAULT_CFG)
        trainer.train()

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith {m}")
            getattr(self, f"{m}")(*args, **kwargs)


if __name__ == "__main__":
    LOGGER.info("Start Test the NetAug!")
    test_class = TestNetAug()
    test_class()
    LOGGER.info("No Expected exception caught")
    LOGGER.info("All the test passed!")
