import torch
import torch.nn as nn
import torch.nn.functional as f
from models.repconv import RepConvBN
from collections import OrderedDict

from typing import List


class GatherExpansion(nn.Module):
    """
    GatherExpansion Module :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, sd_prob: float = 0.0) -> None:
        """
        :param in_channels:  input channels
        :param out_channels: output channels
        :param stride: stride default is 1
        :param sd_prob: stochastic depth probability default is 0.0
        """
        super().__init__()
        self.conv1 = RepConvBN(in_channels, in_channels * 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = RepConvBN(in_channels * 6, in_channels * 6, kernel_size=3, stride=1, padding=1,
                               groups=in_channels * 6)
        self.conv3 = RepConvBN(in_channels * 6, out_channels, kernel_size=1, use_act=False, sd_prob=sd_prob)
        self.convsd = RepConvBN(in_channels * 6, in_channels * 6, kernel_size=3, stride=stride, padding=1,
                                groups=in_channels * 6, use_act=False) if stride != 1 else nn.Identity()
        self.shortcut = nn.Sequential(
            RepConvBN(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      use_act=False),
            RepConvBN(in_channels, out_channels, kernel_size=1, use_act=False)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x_ = self.shortcut(x)
        x = self.conv3(self.conv2(self.convsd(self.conv1(x))))
        return self.act(x + x_)


class Stem(nn.Module):
    """
    BiseNetv2 Stem Module :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self, out_channels: int) -> None:
        """
        :param out_channels: output channels
        """
        super().__init__()
        self.conv1 = RepConvBN(3, out_channels, kernel_size=3, stride=2, padding=1)
        self.branch1 = nn.Sequential(
            RepConvBN(out_channels, out_channels // 2, 1),
            RepConvBN(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
        )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = RepConvBN(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        x = self.conv1(x)
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = self.conv2(x)
        return x


class ContextEmbedding(nn.Module):
    """
    Context Embedding Module :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self, out_channels: int) -> None:
        """
        :param out_channels: output channels
        """
        super().__init__()
        self.avg_conv = nn.Sequential(
            nn.AvgPool2d((1, 1)),
            RepConvBN(out_channels, out_channels, 1)
        )
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        y = self.avg_conv(x)
        x = x + y.expand_as(x)
        return self.conv(x)


class BGA(nn.Module):
    """
    Bilateral Guided Aggregation Module :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self, out_channels: int) -> None:
        """
        :param out_channels: output channels
        """
        super().__init__()
        self.d_branch1 = nn.Sequential(
            RepConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels,
                      use_act=False),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        self.d_branch2 = nn.Sequential(
            RepConvBN(out_channels, out_channels, kernel_size=3, stride=2, padding=1, use_act=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.s_branch1 = nn.Sequential(
            RepConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels,
                      use_act=False),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
        self.s_branch2 = nn.Sequential(
            RepConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_act=False),
            nn.Upsample(scale_factor=4, align_corners=False, mode="bilinear"),
            nn.Sigmoid()
        )
        self.conv = RepConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_act=False)

    def forward(self, semantic_x: torch.Tensor, detail_x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param semantic_x: input feature maps from semantic branch
        :param detail_x: input feature maps from detail branch
        :return: output feature maps
        """
        d = self.d_branch1(detail_x) * self.s_branch2(semantic_x)
        s = self.d_branch2(detail_x) * self.s_branch1(semantic_x)
        s = f.upsample(s, scale_factor=4, align_corners=False, mode="bilinear")
        return self.conv(s + d)


class DetailBranch(nn.Module):
    """
    Detail branch :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage2 = nn.Sequential(
            RepConvBN(3, 64, kernel_size=3, stride=2, padding=1),
            RepConvBN(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.stage4 = nn.Sequential(
            RepConvBN(64, 64, kernel_size=3, stride=2, padding=1),
            RepConvBN(64, 64, kernel_size=3, stride=1, padding=1),
            RepConvBN(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.stage16 = nn.Sequential(
            RepConvBN(64, 128, kernel_size=3, stride=2, padding=1),
            RepConvBN(128, 128, kernel_size=3, stride=1, padding=1),
            RepConvBN(128, 128, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        return self.stage16(self.stage4(self.stage2(x)))


class SemanticBranch(nn.Module):
    """
    Semantic branch :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = Stem(16)
        total_stage_blocks = 8
        sd_probs = torch.arange(0, total_stage_blocks) / total_stage_blocks * 0.1

        self.stage8 = nn.Sequential(
            GatherExpansion(16, 32, stride=2, sd_prob=sd_probs[0]),
            GatherExpansion(32, 32, sd_prob=sd_probs[1]),
        )
        self.stage16 = nn.Sequential(
            GatherExpansion(32, 64, stride=2, sd_prob=sd_probs[2]),
            GatherExpansion(64, 64, sd_prob=sd_probs[3]),
        )
        self.stage32 = nn.Sequential(
            GatherExpansion(64, 128, stride=2, sd_prob=sd_probs[4]),
            GatherExpansion(128, 128, sd_prob=sd_probs[5]),
            GatherExpansion(128, 128, sd_prob=sd_probs[6]),
            GatherExpansion(128, 128, sd_prob=sd_probs[7]),
        )
        self.ce = ContextEmbedding(128)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        outs = OrderedDict()
        x = self.stem(x)
        outs["aux4"] = x
        x = self.stage8(x)
        outs["aux8"] = x
        x = self.stage16(x)
        outs["aux16"] = x
        x = self.stage32(x)
        outs["aux32"] = x
        outs["out"] = self.ce(x)
        return outs


class Head(nn.Module):
    """
    Segmentation head :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self, in_channels: int, num_classes: int, p_drop: float = 0.0) -> None:
        """
        :param in_channels: input channels
        :param num_classes: output channels
        :param p_drop: dropout probability
        """
        super().__init__()
        self.conv = nn.Sequential(
            RepConvBN(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p_drop),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        return self.conv(x)


class BisenetV2(nn.Module):
    """
    BiseNetv2 :)
    Paper: https://arxiv.org/pdf/2004.02147
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.bga = BGA(128)
        self.head = Head(128, num_classes, p_drop=0.2)
        self.aux4 = Head(16, num_classes, p_drop=0.2)
        self.aux8 = Head(32, num_classes, p_drop=0.2)
        self.aux16 = Head(64, num_classes, p_drop=0.2)
        self.aux32 = Head(128, num_classes, p_drop=0.2)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """
        forward function :)
        :param x: input feature maps
        :return: output feature maps
        """
        _, _, H, W = x.shape
        s = self.semantic_branch(x)
        d = self.detail_branch(x)
        s["out"] = self.bga(semantc_x=s["out"], detail_x=d)
        s["out"] = self.head(s["out"])
        s["out"] = f.interpolate(s["out"], size=(H, W), align_corners=False, mode="bilinear")
        if self.training:
            s["aux4"] = f.interpolate(self.aux4(s["aux4"]), size=(H, W), align_corners=False, mode="bilinear")
            s["aux8"] = f.interpolate(self.aux8(s["aux8"]), size=(H, W), align_corners=False, mode="bilinear")
            s["aux16"] = f.interpolate(self.aux16(s["aux16"]), size=(H, W), align_corners=False, mode="bilinear")
            s["aux32"] = f.interpolate(self.aux32(s["aux32"]), size=(H, W), align_corners=False, mode="bilinear")
        else:
            s.pop("aux4")
            s.pop("aux8")
            s.pop("aux16")
            s.pop("aux32")
        return s

    def get_params(self, lr: float, weight_decay: float) -> List:
        """
        get network parameters for optimizer also doesn't weight decay wd for batchnorm :)
        :param lr:
        :param weight_decay:
        :return: list of parameters
        """
        db_wd = []
        db_nwd = []
        sb_wd = []
        sb_nwd = []
        bga_wd = []
        bga_nwd = []
        head_wd = []
        head_nwd = []
        aux4_wd = []
        aux4_nwd = []
        aux8_wd = []
        aux8_nwd = []
        aux16_wd = []
        aux16_nwd = []
        aux32_wd = []
        aux32_nwd = []

        for p in self.detail_branch.parameters():
            if p.dim == 1:
                db_nwd.append(p)
            else:
                db_wd.append(p)

        for p in self.semantic_branch.parameters():
            if p.dim == 1:
                sb_nwd.append(p)
            else:
                sb_wd.append(p)

        for p in self.bga.parameters():
            if p.dim == 1:
                bga_nwd.append(p)
            else:
                bga_wd.append(p)

        for p in self.head.parameters():
            if p.dim == 1:
                head_nwd.append(p)
            else:
                head_wd.append(p)
        for p in self.aux4.parameters():
            if p.dim == 1:
                aux4_nwd.append(p)
            else:
                aux4_wd.append(p)
        for p in self.aux8.parameters():
            if p.dim == 1:
                aux8_nwd.append(p)
            else:
                aux8_wd.append(p)
        for p in self.aux16.parameters():
            if p.dim == 1:
                aux16_nwd.append(p)
            else:
                aux16_wd.append(p)
        for p in self.aux32.parameters():
            if p.dim == 1:
                aux32_nwd.append(p)
            else:
                aux32_wd.append(p)

        params = [
            {"params": db_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": db_nwd, "lr": lr, "weight_decay": 0},

            {"params": sb_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": sb_nwd, "lr": lr, "weight_decay": 0},

            {"params": bga_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": bga_nwd, "lr": lr, "weight_decay": 0},

            {"params": head_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": head_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux4_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux4_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux8_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux8_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux16_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux16_nwd, "lr": lr, "weight_decay": 0},

            {"params": aux32_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": aux32_nwd, "lr": lr, "weight_decay": 0},
        ]
        return params


if __name__ == "__main__":
    pass
