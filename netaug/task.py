import contextlib
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union, Any

import torch
import torch.nn as nn
import netaug

from torch import nn as nn

from ultralytics.nn.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify,
                                    Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Ensemble, Focus,
                                    GhostBottleneck, GhostConv, Segment)
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync,
                                                copy_attr)
from ultralytics.nn.tasks import DetectionModel

from netaug.layers import (DynamicConv2d, DynamicDetect, DynamicSPPF, DynamicConv, DynamicC2f)
from netaug.utils.torch_utils import create_linear_sequence, random_sample


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
    depth, width = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # get module
        if 'nn.' in m:
            m = getattr(torch.nn, m[3:])
        elif 'Dynamic' in m:
            m = getattr(netaug.layers, m)
        else:
            m = globals()[m]

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
                 DynamicConv, DynamicC2f, DynamicSPPF):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x,
                     DynamicC2f):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is DynamicBatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class NetAugDetectionModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True, max_width=1.5, max_depth=1, num_points=3):
        super().__init__(cfg, ch, nc, verbose)
        # For dynamic model attribute
        self.max_width = max_width
        self.max_depth = max_depth
        self.ch = ch
        self.aug_width = create_linear_sequence(1.0, self.max_width, num_points=num_points) / self.max_width
        self.model, self.save = self.parse_dynamic_model(deepcopy(self.yaml), ch=ch, verbose=verbose)

        m = self.model[-1]  # Detect()
        if isinstance(m, (DynamicDetect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def parse_dynamic_model(self, d, ch, verbose=True):
        import ast

        # Args
        max_channels = float('inf')
        nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
        depth, width = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple'))
        if scales:
            scale = d.get('scale')
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")  # print

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, ((f, n, _, args), m) in enumerate(
                zip(d['backbone'] + d['head'], self.model)):  # from, number, module, args
            if "Dynamic" + m.__class__.__name__ in netaug.layers.__all__:
                m = getattr(netaug.layers, "Dynamic" + m.__class__.__name__)
            else:
                class_name = m.__class__.__name__
                if class_name in torch.nn.modules.__all__:
                    m = getattr(torch.nn, class_name)
                else:
                    m = globals()[class_name]

            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

            if m in (DynamicConv, DynamicSPPF, DynamicC2f):
                c1, c2 = ch[f], args[0]
                if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                    c2 = make_divisible(make_divisible(c2 * width, 8) * self.max_width, 1)

                args = [c1, c2, *args[1:]]
                if m in (DynamicC2f,):
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in (DynamicDetect, Segment):
                args.append([ch[x] for x in f])
                if m is Segment:
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            m.np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            if verbose:
                LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    def set_active(self, aug_width):
        aug_width = aug_width.unsqueeze(0) if aug_width.shape == torch.Size([]) else aug_width
        ch = [self.ch]  # record all output
        c2 = ch[-1]  # layers, savelist, ch out
        for m in self.model:
            i, f = m.i, m.f
            width_scaler = random_choices(aug_width)
            if isinstance(m, (DynamicConv, DynamicC2f, DynamicSPPF)):
                c1, c2 = ch[f], int(m.get_out_channels * width_scaler)
                m.set_active(c1, c2)
            elif isinstance(m, (Concat,)):
                c2 = sum(ch[x] for x in f)
            elif isinstance(m, DynamicDetect):
                ch_list = [ch[x] for x in f]
                m.set_active(ch_list)
            if i == 0:
                ch = []
            ch.append(c2)
        return ch

    def export_module(self, verbose=True) -> DetectionModel:
        module = DetectionModel.__new__(DetectionModel)
        nn.Module.__init__(module)
        copy_attr(module, self, exclude=("model", "max_width", "max_depth", "ch", "aug_width"))
        # set active
        self.set_active(self.aug_width[0])
        export_m = []
        for m in self.model:
            if hasattr(m, "export_module"):
                export_m.append(m.export_module())
            else:
                export_m.append(m)
        module.model = nn.Sequential(*export_m)
        m = module.model[-1]  # Detect()

        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, self.ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(module)
        if verbose:
            module.info()
            LOGGER.info('')
        return module


def random_choices(src_list: List[Any], generator: Optional[torch.Generator] = None, k=1) -> Union[Any, List[Any]]:
    rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
    out_list = [src_list[i] for i in rand_idx]
    return out_list[0] if k == 1 else out_list
