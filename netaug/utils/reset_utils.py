import numpy as np
import torch
import torch.nn as nn

from typing import List, Optional, Union
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from copy import deepcopy
from tqdm import tqdm
from ultralytics.yolo.utils import (LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save, DEFAULT_CFG)


class AverageMeter(object):
    """Computes and stores the average and current value.

    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[torch.Tensor, np.ndarray, float, int], n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

