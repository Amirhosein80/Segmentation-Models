import torch
import torch.nn as nn
import torchvision.ops as ops
from model_utils import SEBlock
from typing import Tuple


def _conv_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1,
             sd_prob=0.0) -> nn.Sequential:
    """
    Conv + BatchNormalization :)
    :param in_channels: input channels
    :param out_channels: output channels
    :param kernel_size: kernel size default is 3
    :param padding: number of padding in each side default is 1
    :param stride: kernel stride default is 1
    :param dilation: dilation default is 1
    :param groups: number of groups default is 1
    :param sd_prob: stochastic depth probability default is 0.0
    :return: sequential of conv & batchnorm
    """
    block = nn.Sequential()
    block.add_module("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                       dilation=dilation, bias=False)
                     )
    block.add_module("bn", nn.BatchNorm2d(num_features=out_channels)
                     )
    block.add_module("sd", ops.StochasticDepth(sd_prob, mode="row")
                     )
    return block


def _fuse_conv_bn(m: nn.Sequential) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert conv & batchnorm weights to conv weights & bias :)
    :param m: input sequential
    :return: kernel & bias
    """
    kernel = m.conv.weight
    mean = m.bn.running_mean
    var = m.bn.running_var
    gamma = m.bn.weight
    beta = m.bn.bias
    eps = m.bn.eps
    std = (var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    kernel = kernel * t
    bias = beta - (mean * gamma / std)
    return kernel, bias


class DWShortCut(nn.Module):
    """
    IDEA: Use a separable conv with stride instead of conv1x1 with stride :)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                 stride: int = 1, groups: int = 1, dilation: int = 1, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size default is 3
        :param padding: number of padding in each side default is 1
        :param stride: kernel stride default is 1
        :param dilation: dilation default is 1
        :param groups: number of groups default is 1
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.convdw = _conv_bn(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=in_channels, dilation=dilation)
        self.convpw = _conv_bn(in_channels, out_channels, 1, stride=1, padding=0, groups=groups, sd_prob=sd_prob)

    def forward(self, x) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x = self.convdw(x)
        return self.convpw(x)

    def fuse_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        convert separable conv weights to weight & bias :)
        :return: kernel & bias
        """
        kernel_dw, bias_dw = _fuse_conv_bn(self.convdw)
        kernel_pw, bias_pw = _fuse_conv_bn(self.convpw)
        kernel = []
        bias = []
        for kd, kp, bd in zip(torch.tensor_split(kernel_dw, self.groups, 0),
                              torch.tensor_split(kernel_pw, self.groups, 0),
                              torch.tensor_split(bias_dw, self.groups, 0)):
            k = kd.permute(1, 0, 2, 3) * kp.expand(kp.shape[0], kp.shape[1], self.kernel_size, self.kernel_size)
            b = bd @ kp.permute(-1, -2, -3, -4)
            kernel.append(k)
            bias.append(b)
        kernel = torch.cat(kernel, dim=0)
        bias = torch.cat(bias, dim=-1)
        bias = bias.reshape(-1) + bias_pw
        return kernel, bias


class RepConvBN(nn.Module):
    """
    Parameterizable conv & batchnorm with 2 ideas :)
    First IDEA: add depthwise conv to extract only spatial features :)
    Second IDEA: use a separable conv with stride instead of conv1x1 with stride :)
    Reference Paper: https://arxiv.org/pdf/2101.03697
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, groups: int = 1, dilation: int = 1, use_act: bool = True, use_attn: bool = False,
                 infer_mode: bool = False, use_channel_scaler: bool = False, use_skip: bool = False,
                 use_spatial_scaler: bool = False, sd_prob: float = 0.0):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size default is 3
        :param padding: number of padding in each side default is 1
        :param stride: kernel stride default is 1
        :param dilation: dilation default is 1
        :param groups: number of groups default is 1
        :param use_act: use activation function default is True
        :param use_attn: use self attention module (SENT) default is False
        :param infer_mode: inference mode default is False
        :param use_channel_scaler: use channel scaler (conv 1x1 in RepVGG Block) if stride !=1 use separable conv default is False
        :param use_skip: use batchnorm skip connection if stride !=1 doesn't use skip connection default is False
        :param use_spatial_scaler: use depthwise conv for extract spatial features default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.sd_prob = sd_prob
        self.use_channel_scaler = use_channel_scaler
        self.use_spatial_scaler = use_spatial_scaler
        self.use_act = use_act
        self.use_attn = use_attn
        self.use_skip = use_skip
        self.infer_mode = infer_mode

        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()
        self.attn = SEBlock(out_channels) if use_attn else nn.Identity()

        if infer_mode:
            self.rep_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            self.conv = _conv_bn(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                                 dilation=dilation, groups=groups, sd_prob=sd_prob)
            self.scale_conv = _conv_bn(in_channels, out_channels, kernel_size=1, stride=1, dilation=1,
                                       padding=0, groups=groups,
                                       sd_prob=sd_prob) if use_channel_scaler and kernel_size > 1 else None
            if stride != 1 and self.scale_conv is not None:
                self.scale_conv = DWShortCut(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation, sd_prob=sd_prob, groups=groups)
            self.spatial_conv = _conv_bn(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                         stride=stride, dilation=dilation, sd_prob=sd_prob,
                                         groups=out_channels) if use_spatial_scaler and in_channels == out_channels \
                                                                 and kernel_size > 1 and groups != out_channels else None
            self.skip_bn = nn.BatchNorm2d(num_features=in_channels) if use_skip and in_channels == out_channels and \
                                                                       stride == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        if self.infer_mode:
            return self.act(self.attn(self.rep_conv(x)))
        out = 0
        if self.scale_conv is not None: out += self.scale_conv(x)
        if self.spatial_conv is not None: out += self.spatial_conv(x)
        if self.skip_bn is not None: out += self.skip_bn(x)

        out += self.conv(x)
        return self.act(self.attn(out))

    def reparameterize(self) -> None:
        """
        convert all module to a conv
        """
        if self.infer_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.rep_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=self.groups, bias=True)
        self.rep_conv.weight.data = kernel
        self.rep_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('scale_conv')
        self.__delattr__('skip_bn')

        self.infer_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        convert all module weights to weights & bias of a conv2d
        :return: weights & bias
        """
        kernel_conv, bias_conv = self._fuse_conv()
        kernels = [kernel_conv]
        biases = [bias_conv]
        if self.scale_conv is not None:
            kernel_scale, bias_scale = self._fuse_scale_conv()

            kernels.append(kernel_scale)
            biases.append(bias_scale)

        if self.skip_bn is not None:
            kernel_bn, bias_bn = self._fuse_skip_bn()
            kernels.append(kernel_bn)
            biases.append(bias_bn)

        if self.spatial_conv is not None:
            kernel_spatial, bias_spatial = self._fuse_spatial_conv()
            kernels.append(kernel_spatial)
            biases.append(bias_spatial)

        return sum(kernels), sum(biases)

    def _fuse_skip_bn(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        convert skip connection to weights & bias
        :return: weights & bias
        """
        input_dim = self.in_channels // self.groups
        kernel = torch.zeros((self.in_channels,
                              input_dim,
                              self.kernel_size,
                              self.kernel_size),
                             dtype=self.conv.conv.weight.dtype,
                             device=self.conv.conv.weight.device)
        for i in range(self.in_channels):
            kernel[i, i % input_dim,
                   self.kernel_size // 2,
                   self.kernel_size // 2] = 1
        running_mean = self.skip_bn.running_mean
        running_var = self.skip_bn.running_var
        gamma = self.skip_bn.weight
        beta = self.skip_bn.bias
        eps = self.skip_bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        kernel = kernel * t
        bias = beta - running_mean * gamma / std
        return kernel, bias

    def _fuse_spatial_conv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        convert depthwise conv to weights & bias
        :return: weights & bias
        """
        spatial_kernel, spatial_bias = _fuse_conv_bn(self.spatial_conv)
        input_dim = self.in_channels // self.groups
        kernel = torch.zeros((self.in_channels,
                              input_dim,
                              self.kernel_size,
                              self.kernel_size),
                             dtype=self.conv.conv.weight.dtype,
                             device=self.conv.conv.weight.device)
        for i in range(self.in_channels):
            kernel[i, i % input_dim] = spatial_kernel[i, 0]

        return kernel, spatial_bias

    def _fuse_scale_conv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        convert conv1x1 + bn to weights & bias
        :return: weights & bias
        """
        if isinstance(self.scale_conv, nn.Sequential):
            kernel_scale, bias_scale = _fuse_conv_bn(self.scale_conv)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])
        else:
            kernel_scale, bias_scale = self.scale_conv.fuse_weights()
        return kernel_scale, bias_scale

    def _fuse_conv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        convert conv + bn to weights & bias
        :return: weights & bias
        """
        return _fuse_conv_bn(self.conv)


if __name__ == "__main__":
    pass
