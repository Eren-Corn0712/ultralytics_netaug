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


#
# class DetectionAugModel(BaseModel):
#     # YOLOv8 detection model
#     def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
#         super().__init__()
#         self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
#
#         # Define model
#         ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
#         if nc and nc != self.yaml['nc']:
#             LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
#             self.yaml['nc'] = nc  # override yaml value
#         self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
#         self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
#         self.inplace = self.yaml.get('inplace', True)
#
#         # Build strides
#         m = self.model[-1]  # Detect()
#         if isinstance(m, (Detect, Segment)):
#             s = 256  # 2x min stride
#             m.inplace = self.inplace
#             forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
#             m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
#             self.stride = m.stride
#             m.bias_init()  # only run once
#
#         # Init weights, biases
#         initialize_weights(self)
#         if verbose:
#             self.info()
#             LOGGER.info('')
#
#         # Init NetAug Attribute
#         self.ch = ch
#
#     def forward(self, x, augment=False, profile=False, visualize=False):
#         if augment:
#             return self._forward_augment(x)  # augmented inference, None
#         return self._forward_once(x, profile, visualize)  # single-scale inference, train
#
#     def _forward_augment(self, x):
#         img_size = x.shape[-2:]  # height, width
#         s = [1, 0.83, 0.67]  # scales
#         f = [None, 3, None]  # flips (2-ud, 3-lr)
#         y = []  # outputs
#         for si, fi in zip(s, f):
#             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
#             yi = self._forward_once(xi)[0]  # forward
#             # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
#             yi = self._descale_pred(yi, fi, si, img_size)
#             y.append(yi)
#         y = self._clip_augmented(y)  # clip augmented tails
#         return torch.cat(y, -1), None  # augmented inference, train
#
#     @staticmethod
#     def _descale_pred(p, flips, scale, img_size, dim=1):
#         # de-scale predictions following augmented inference (inverse operation)
#         p[:, :4] /= scale  # de-scale
#         x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
#         if flips == 2:
#             y = img_size[0] - y  # de-flip ud
#         elif flips == 3:
#             x = img_size[1] - x  # de-flip lr
#         return torch.cat((x, y, wh, cls), dim)
#
#     def _clip_augmented(self, y):
#         # Clip YOLOv5 augmented inference tails
#         nl = self.model[-1].nl  # number of detection layers (P3-P5)
#         g = sum(4 ** x for x in range(nl))  # grid points
#         e = 1  # exclude layer count
#         i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
#         y[0] = y[0][..., :-i]  # large
#         i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
#         y[-1] = y[-1][..., i:]  # small
#         return y
#
#     def set_active(self, width_scalers):  # Detect()
#         head_idx = self.model[-1].f
#         out_ch = [self.ch]  # record all output
#         for m in self.model[:-1]:
#             input_ch = out_ch[-1]
#             if m.f != -1:  # if not from previous layer
#                 input_ch = [out_ch[j] if j == -1 else out_ch[j + 1] for j in m.f]
#                 input_ch = sum(input_ch) if isinstance(input_ch, list) else input_ch
#
#             if isinstance(m, (DynamicConv, DynamicC2f, DynamicSPPF)):
#                 width_scaler = random_choices(width_scalers)
#                 c_out = m.get_out_channels() if m.i in head_idx else int(width_scaler * m.get_out_channels())
#                 m.set_active(input_ch, c_out)
#                 out_ch.append(c_out)
#
#             if isinstance(m, nn.Sequential) and m.i not in head_idx:
#                 inner_ch = [input_ch]
#                 for n in m:
#                     if isinstance(n, (DynamicConv, DynamicC2f, DynamicSPPF)):
#                         width_scaler = random_choices(width_scalers)
#                         c_out = int(width_scaler * n.get_out_channels())
#                         n.set_active(inner_ch[-1], c_out)
#                         inner_ch.append(c_out)
#                 out_ch.append(inner_ch[-1])
#
#             if isinstance(m, (Concat, nn.Upsample)):
#                 out_ch.append(input_ch)
#
#         return out_ch


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

    def export_module(self) -> DetectionModel:
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
        return module


def random_choices(src_list: List[Any], generator: Optional[torch.Generator] = None, k=1) -> Union[Any, List[Any]]:
    rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
    out_list = [src_list[i] for i in rand_idx]
    return out_list[0] if k == 1 else out_list
