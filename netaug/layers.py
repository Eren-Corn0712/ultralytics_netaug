import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Optional, Tuple, Union
from ultralytics.nn.modules import autopad, Conv, Bottleneck, C2f, SPPF, Detect, DFL
from ultralytics.yolo.utils.torch_utils import copy_attr
from netaug.utils.export_utils import get_parent_class_name
from netaug.utils.torch_utils import sort_param, calc_importance

__all__ = [
    'DynamicConv2d',
    'DynamicBatchNorm2d',
    'DynamicBottleneck',
    'DynamicConv',
    'DynamicC2f',
    'DynamicSPPF',
    'DynamicDetect',
]


class DynamicModule(nn.Module):
    """
    A custom PyTorch module that serves as a base class for modules with dynamic architectures.
    """

    def export_module(self) -> nn.Module:
        """
        This method should be overridden by child classes to return a module representing the current
        active architecture of the dynamic module.
        """
        raise NotImplementedError

    def active_state_dict(self):
        """
        Recursively returns a dictionary containing the state of the module and its active sub-modules.

        Returns:
            A dictionary containing the state of the module and its active sub-modules.
        """
        state_dict = self.state_dict()  # get the state dictionary of the current module
        for prefix, module in self.named_children():  # iterate over all sub-modules of the current module
            if isinstance(module,
                          DynamicModule):  # if the sub-module is a dynamic module, recursively get its active state
                for name, tensor in module.active_state_dict().items():
                    state_dict[prefix + "." + name] = tensor
        return state_dict


class DynamicConv2d(DynamicModule, nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            groups, bias, padding_mode, **factory_kwargs)

        self.active_in_channels = in_channels
        self.active_out_channels = out_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_channels].contiguous()

    def forward(self, input: Tensor) -> Tensor:
        self.active_in_channels = input.shape[1]
        return self._conv_forward(input, self.active_weight, self.active_bias)

    def export_module(self) -> nn.Module:
        module = nn.Conv2d(
            self.active_in_channels, self.active_out_channels,
            self.kernel_size, self.stride, self.padding, self.dilation, self.groups,
            self.bias is not None, self.padding_mode
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self):
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict

    def sort_channels(self):
        # equal torch.sum(torch.abs(self.conv.weight.data), dim=(0, 2, 3))
        # sorted weight by compute the channels.
        # nn.Conv2d weight size is [c_out,c_in,k_size,k_size]
        # nn.Conv2d bias weight size is [c_out]
        sorted_idx = calc_importance(self.weight, dim=(0, 2, 3))
        if self.weight is not None:
            sort_param(self.weight, dim=1, sorted_idx=sorted_idx)  # weight
        if self.bias is not None:
            sort_param(self.bias, dim=0, sorted_idx=sorted_idx)  # bias


class DynamicBatchNorm2d(DynamicModule, nn.BatchNorm2d):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.BatchNorm2d.__init__(
            self, num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.active_num_features = num_features

    @property
    def active_running_mean(self) -> Optional[torch.Tensor]:
        if self.running_mean is None:
            return None
        return self.running_mean[: self.active_num_features]

    @property
    def active_running_var(self) -> Optional[torch.Tensor]:
        if self.running_var is None:
            return None
        return self.running_var[: self.active_num_features]

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[: self.active_num_features]

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_num_features]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        self.active_num_features = input.shape[1]

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
          Decide whether the mini-batch stats should be used for normalization rather than the buffers.
          Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.active_running_mean is None) and (self.active_running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            self.active_running_mean if not self.training or self.track_running_stats else None,
            self.active_running_var if not self.training or self.track_running_stats else None,
            self.active_weight,
            self.active_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def export_module(self) -> nn.Module:
        module = nn.BatchNorm2d(
            self.active_num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self):
        state_dict = super().active_state_dict()
        if self.running_mean is not None:
            state_dict["running_mean"] = self.active_running_mean
        if self.running_var is not None:
            state_dict["running_var"] = self.active_running_var
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict

    def sort_channels(self):
        # nn.BatchNorm2d weight size is (num_feature)
        # nn.BatchNorm2d bias size is (num_feature)
        sorted_idx = calc_importance(self.weight, dim=0)
        if self.weight is not None:
            sort_param(self.weight, dim=0, sorted_idx=sorted_idx)
        if self.bias is not None:
            sort_param(self.bias, dim=0, sorted_idx=sorted_idx)

        if self.running_mean is not None:
            sort_param(self.running_mean, dim=0, sorted_idx=sorted_idx)

        if self.running_var is not None:
            sort_param(self.running_var, dim=0, sorted_idx=sorted_idx)

class DynamicConv(Conv, DynamicModule):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        nn.Module.__init__(self)
        self.conv = DynamicConv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = DynamicBatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def export_module(self) -> Conv:
        module = Conv.__new__(Conv)
        nn.Module.__init__(module)
        copy_attr(module, self, exclude=("conv", "bn", "act"))
        module.conv = self.conv.export_module()
        module.bn = self.bn.export_module()
        module.act = self.act
        # Convert the netaug.layers to ultralytics.nn.layers
        module.type = get_parent_class_name(self)
        return module

    def set_active(self, c_in, c_out):
        self.conv.active_in_channels = c_in
        self.conv.active_out_channels = c_out
        self.bn.active_num_features = c_out

    def sort_channels(self):
        self.conv.sort_channels()
        self.bn.sort_channels()

    @property
    def get_out_channels(self):
        return self.conv.out_channels


class DynamicBottleneck(Bottleneck, DynamicModule):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        nn.Module.__init__(self)
        self.c_ = int(c2 * e)  # hidden channels
        self.cv1 = DynamicConv(c1, self.c_, k[0], 1)
        self.cv2 = DynamicConv(self.c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.shortcut = shortcut
        self.e = e

    def export_module(self) -> Bottleneck:
        module = Bottleneck.__new__(Bottleneck)
        nn.Module.__init__(module)
        copy_attr(module, self, exclude=("cv1", "cv2", "add", "e", "shortcut", "c_"))
        module.cv1 = self.cv1.export_module()
        module.cv2 = self.cv2.export_module()
        module.add = self.add
        module.type = get_parent_class_name(self)
        return module

    def set_active(self, c1, c2):
        c_ = int(c2 * self.e)
        self.cv1.set_active(c1, c_)
        self.cv2.set_active(c_, c2)
        self.add = self.shortcut and c1 == c2

    def sort_channels(self):
        self.cv1.sort_channels()
        self.cv2.sort_channels()

    @property
    def get_out_channels(self):
        return self.cv2.conv.out_channels


class DynamicC2f(C2f, DynamicModule):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        nn.Module.__init__(self)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = DynamicConv(c1, 2 * self.c, 1, 1)
        self.cv2 = DynamicConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            DynamicBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.n = n
        self.e = e

    def export_module(self) -> C2f:
        module = C2f.__new__(C2f)
        nn.Module.__init__(module)
        copy_attr(module, self, exclude=("cv1", "cv2", "m", "n", "e"))
        module.cv1 = self.cv1.export_module()
        module.cv2 = self.cv2.export_module()
        module.m = nn.ModuleList([m.export_module() for m in self.m])
        module.type = get_parent_class_name(self)
        return module

    def sort_channels(self):
        self.cv1.sort_channels()
        self.cv2.sort_channels()
        [m.set_active() for m in self.m]

    def set_active(self, c1, c2):
        c = int(c2 * self.e)
        self.cv1.set_active(c1, 2 * c)
        self.cv2.set_active((2 + self.n) * c, c2)
        for m in self.m:
            m.set_active(c, c)

    def sort_channels(self):
        pass

    @property
    def get_out_channels(self):
        return self.cv2.conv.out_channels


class DynamicSPPF(SPPF, DynamicModule):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13)):
        nn.Module.__init__(self)
        self.c_ = c1 // 2  # hidden channels
        self.cv1 = DynamicConv(c1, self.c_, 1, 1)
        self.cv2 = DynamicConv(self.c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def set_active(self, c1, c2):
        c_ = c1 // 2
        self.cv1.set_active(c1, c_)
        self.cv2.set_active(c_ * 4, c2)

    def sort_channels(self):
        self.cv1.sort_channels()
        self.cv2.sort_channels()

    def export_module(self) -> SPPF:
        module = SPPF.__new__(SPPF)
        nn.Module.__init__(module)
        copy_attr(module, self, exclude=("cv1", "cv2", "m", "c_"))
        module.cv1 = self.cv1.export_module()
        module.cv2 = self.cv2.export_module()
        module.m = self.m
        module.type = get_parent_class_name(self)
        return module

    @property
    def get_out_channels(self):
        return self.cv2.conv.out_channels


class DynamicDetect(Detect, DynamicModule):
    def __init__(self, nc=80, ch=()):  # detection layer
        nn.Module.__init__(self)
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DynamicConv(x, c2, 3), DynamicConv(c2, c2, 3), DynamicConv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                DynamicConv(x, c3, 3), DynamicConv(c3, c3, 3), DynamicConv2d(c3, self.nc, 1)
            ) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def set_active(self, c1_list):
        c2, c3 = max((16, c1_list[0] // 4, self.reg_max * 4)), max(c1_list[0], self.nc)  # channels
        for m, c1 in zip(self.cv2, c1_list):
            m[0].set_active(c1, c2)
            m[1].set_active(c2, c2)
            m[2].active_in_channels = c2
            m[2].active_out_channels = 4 * self.reg_max

        for m, c1 in zip(self.cv3, c1_list):
            m[0].set_active(c1, c3)
            m[1].set_active(c3, c3)
            m[2].active_in_channels = c3
            m[2].active_out_channels = self.nc

    def sort_channels(self):
        for m in self.cv2:
            for n in m:
                n.sort_channels()

        for m in self.cv3:
            for n in m:
                n.sort_channels()

    def export_module(self) -> Detect:
        module = Detect.__new__(Detect)
        nn.Module.__init__(module)
        copy_attr(module, self, exclude=("cv2", "cv3", "dfl"))
        module.cv2 = nn.ModuleList(
            nn.Sequential(x[0].export_module(), x[1].export_module(), x[2].export_module()) for x in self.cv2)

        module.cv3 = nn.ModuleList(
            nn.Sequential(x[0].export_module(), x[1].export_module(), x[2].export_module()) for x in self.cv3)

        module.dfl = self.dfl
        module.type = get_parent_class_name(self)
        return module
