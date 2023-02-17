import copy
import math

import torch
import torch.nn as nn

from typing import Optional


# ============
#  Functions
# =============


def efficientnet_init_weights(m: nn.Module) -> None:
    """
    EfficientNet weight initialization :)
    Reference Paper: https://arxiv.org/pdf/1905.11946
    :param m: module
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)


def regnet_init_weights(m: nn.Module) -> None:
    """
    RegNet weight initialization :)
    Reference Paper: https://arxiv.org/pdf/2003.13678
    :param m: module
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out = fan_out // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def convnext_init_weights(m: nn.Module) -> None:
    """
    ConvNext weight initialization :)
    Reference Paper: https://arxiv.org/pdf/2201.03545
    :param m: module
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def set_bn_momentum(model, momentum=0.1) -> None:
    """
    change batch norm momentum in a model :)
    :param model: model
    :param momentum: new momentum
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """
    reparameterize model if model have repconv :)
    :param model: model
    :return: reparameterize model
    """
    rep_model = copy.deepcopy(model)
    for module in rep_model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
        for para in module.parameters():
            para.detach_()
    return rep_model


def replace_relu_with_silu(module) -> None:
    """
    replace relu in a model with silu :)
    :param module:
    :return:
    """
    for name, m in module.named_children():
        if isinstance(m, nn.ReLU):
            setattr(module, name, nn.SiLU())


def replace_relu_with_gelu(module) -> None:
    """
    replace relu in a model with gelu :)
    :param module:
    :return:
    """
    for name, m in module.named_children():
        if isinstance(m, nn.ReLU):
            setattr(module, name, nn.GELU())


# ============
#  Classes
# =============


class EcaModule(nn.Module):
    """
    Efficient Channel Attention for Deep Convolutional Neural Networks :)
    Reference Paper: https://arxiv.org/pdf/1910.03151
    """

    def __init__(self, channels: Optional[int] = None, kernel_size: int = 3, gamma: int = 2, beta: int = 1):
        """
        :param channels: number of channels
        :param kernel_size: kernel size default is 3
        :param gamma: gamma default is 2
        :param beta: beta default is 1
        """
        super().__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        else:
            kernel_size = kernel_size
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Networks :)
    Reference Paper: https://arxiv.org/pdf/1910.03151
    """

    def __init__(self, channels: int, scale: float = 0.25) -> None:
        """
        :param channels: number of channels
        :param scale: a scale for calculate mid-channels (mid-channels = channels * scale) default is 0.25
        """
        super().__init__()
        self.reduce = nn.Conv2d(in_channels=channels, out_channels=int(channels * scale), kernel_size=1,
                                stride=1, bias=True)
        self.relu = nn.ReLU()
        self.expand = nn.Conv2d(in_channels=int(channels * scale), out_channels=channels, kernel_size=1,
                                stride=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.avg_pool(x)
        x_ = self.reduce(x_)
        x_ = self.relu(x_)
        x_ = self.expand(x_)
        x_ = torch.sigmoid(x_)
        return x * x_.expand_as(x)


class EffectiveSEBlock(nn.Module):
    """
    Effective Squeeze-and-Excitation Networks :)
    Reference Paper: https://arxiv.org/pdf/1911.06667v6
    """

    def __init__(self, channels, add_maxpool: bool = False):
        """
        :param channels: number of channels
        :param add_maxpool: use max pool + avg pool. default is False
        """
        super().__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


if __name__ == "__main__":
    pass
