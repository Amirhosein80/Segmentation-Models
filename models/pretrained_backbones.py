import torch
import torch.nn as nn
import torchinfo
import torchvision.models as models
from collections import OrderedDict

"""
    Models 
    1. EfficientNet B0
"""


def _switch_to_dilated_conv3(m):
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (3, 3):
            if m.stride != 1:
                m.stride = 1
            else:
                m.dilation = 2
                m.padding = 2

        if m.kernel_size == (5, 5):
            if m.stride != 1:
                m.stride = 1
            else:
                m.dilation = 2
                m.padding = 4


class EfficientNet(nn.Module):
    """
    EfficientNet B0 Backbone with dilated conv in last stage :)
    Paper: https://arxiv.org/pdf/1905.11946
    """

    def __init__(self) -> None:
        super().__init__()
        self.gradients = []

        features = models.efficientnet_b0(models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        self.stem = nn.Sequential(*features[0: 2])
        self.layer1 = nn.Sequential(*features[2:3])
        self.layer2 = nn.Sequential(*features[3:4])
        self.layer3 = nn.Sequential(*features[4:6])
        self.layer4 = nn.Sequential(*features[6:8])
        self.layer4.apply(_switch_to_dilated_conv3)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """
        Forward function :)
        :param x: input tensor
        :return: output dictionary for feature maps
        """
        out = OrderedDict()
        x = self.stem(x)

        x = self.layer1(x)
        out["1"] = x

        if self.training:
            h = x.register_hook(self.activations_hook)

        x = self.layer2(x)
        out["2"] = x
        if self.training:
            h = x.register_hook(self.activations_hook)

        x = self.layer3(x)
        x = self.layer4(x)
        out["3"] = x
        if self.training:
            h = x.register_hook(self.activations_hook)

        return out

    def activations_hook(self, grad):
        self.gradients.append(grad)

    def get_activations_gradient(self):
        return self.gradients_1, self.gradients_2, self.gradients_3

    def get_activations(self, x):
        return self.forward(x)


if __name__ == "__main__":
    model = EfficientNet()
    print(torchinfo.summary(model, (1, 3, 224, 224)))
