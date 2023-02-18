import torch
import torch.nn as nn
from models.repconv import RepConvBN


class ResNetBlock(nn.Module):
    """
    ResNet Block :)
    Paper: https://arxiv.org/pdf/1512.03385
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 0.25,
                 use_attn: bool = False, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param expand_ratio: expand ratio to calculate mid-channels default is 0.25
        :param use_attn: use self attention module default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        mid_channels = int(expand_ratio * out_channels)
        self.conv1 = RepConvBN(in_channels, mid_channels, 1)
        self.conv2 = RepConvBN(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = RepConvBN(mid_channels, out_channels, 1, use_act=False, use_attn=use_attn, sd_prob=sd_prob)
        self.short_cut = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.short_cut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                RepConvBN(in_channels, out_channels, 1, use_act=False),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.short_cut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + x_)


class ResNextBlock(nn.Module):
    """
    ResNext Block :)
    Paper: https://arxiv.org/pdf/1611.05431
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 0.5,
                 use_attn: bool = False, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param expand_ratio: expand ratio to calculate mid-channels default is 0.5
        :param use_attn: use self attention module default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        mid_channels = int(expand_ratio * out_channels)
        self.conv1 = RepConvBN(in_channels, mid_channels, 1)
        self.conv2 = RepConvBN(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=32)
        self.conv3 = RepConvBN(mid_channels, out_channels, 1, use_act=False, use_attn=use_attn, sd_prob=sd_prob)
        self.short_cut = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.short_cut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                RepConvBN(in_channels, out_channels, 1, use_act=False),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.short_cut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + x_)


class RegNetBlock(nn.Module):
    """
    RegNet Block :)
    Paper: https://arxiv.org/pdf/2003.13678
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 1,
                 use_attn: bool = False, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param expand_ratio: expand ratio to calculate mid-channels default is 1
        :param use_attn: use self attention module default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        mid_channels = int(expand_ratio * out_channels)
        self.conv1 = RepConvBN(in_channels, mid_channels, 1)
        self.conv2 = RepConvBN(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
                               groups=mid_channels // 16)
        self.conv3 = RepConvBN(mid_channels, out_channels, 1, use_act=False, use_attn=use_attn, sd_prob=sd_prob)
        self.short_cut = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.short_cut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                RepConvBN(in_channels, out_channels, 1, use_act=False),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.short_cut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + x_)


class InVResBlock(nn.Module):
    """
    Inverted Residual Block :)
    Paper: https://arxiv.org/pdf/1801.04381
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 6,
                 use_attn: bool = False, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param expand_ratio: expand ratio to calculate mid-channels default is 6
        :param use_attn: use self attention module default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        mid_channels = int(expand_ratio * out_channels)
        self.conv1 = RepConvBN(in_channels, mid_channels, 1)
        self.conv2 = RepConvBN(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
                               groups=mid_channels)
        self.conv3 = RepConvBN(mid_channels, out_channels, 1, use_act=False, use_attn=use_attn, sd_prob=sd_prob)
        self.short_cut = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.short_cut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                RepConvBN(in_channels, out_channels, 1, use_act=False),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.short_cut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + x_)


class FuseIRBlock(nn.Module):
    """
    Fused Inverted Residual Block :)
    Paper: https://arxiv.org/pdf/2104.00298
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 4,
                 use_attn: bool = False, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param expand_ratio: expand ratio to calculate mid-channels default is 4
        :param use_attn: use self attention module default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        mid_channels = int(expand_ratio * out_channels)
        self.conv1 = RepConvBN(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = RepConvBN(mid_channels, out_channels, 1, use_act=False, use_attn=use_attn, sd_prob=sd_prob)
        self.short_cut = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.short_cut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                RepConvBN(in_channels, out_channels, 1, use_act=False),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.short_cut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + x_)


class DarkBlock(nn.Module):
    """
    DarkNet Block :)
    Paper: https://arxiv.org/pdf/1804.02767
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 0.5,
                 use_attn: bool = False, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param expand_ratio: expand ratio to calculate mid-channels default is 0.5
        :param use_attn: use self attention module default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        mid_channels = int(expand_ratio * out_channels)
        self.conv1 = RepConvBN(in_channels, mid_channels, 1, use_act=True)
        self.conv2 = RepConvBN(mid_channels, out_channels, kernel_size=3, stride=stride, padding=1, use_attn=use_attn,
                               sd_prob=sd_prob)
        self.short_cut = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.short_cut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                RepConvBN(in_channels, out_channels, 1, use_act=False),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.short_cut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + x_)


class RepResNetBlock(nn.Module):
    """
    RepResNet Block :)
    Paper: https://arxiv.org/pdf/2203.16250
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: float = 0.5,
                 use_attn: bool = False, sd_prob: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param expand_ratio: expand ratio to calculate mid-channels default is 0.5
        :param use_attn: use self attention module default is False
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        mid_channels = int(expand_ratio * out_channels)
        self.conv1 = RepConvBN(in_channels, mid_channels, 3, use_act=True, stride=stride, padding=1)
        self.conv2 = RepConvBN(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, use_channel_scaler=True,
                               use_spatial_scaler=True, use_attn=use_attn, sd_prob=sd_prob)
        self.short_cut = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.short_cut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                RepConvBN(in_channels, out_channels, 1, use_act=False),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.short_cut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + x_)
