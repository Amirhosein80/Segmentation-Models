import math
from typing import Dict, List, Tuple, Union, Any

import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import InterpolationMode


def _apply_op(
        img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode,
        fill: Union[int, List[int]]
):
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


def _augmentation_space_rand_aug(num_bins: int, image_size: Tuple[int, int]) -> tuple[
    dict[str | Any, tuple[Tensor, bool] | Any], list[str]]:
    affine_ops = [
        "Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"
    ]
    aug_space = {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
        "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
        "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
    }
    return aug_space, affine_ops


def _augmentation_space_tri_aug(num_bins: int) -> tuple[
    dict[str | Any, tuple[Tensor, bool] | Any], list[str]]:
    aug_space = {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
        "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
        "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
        "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
        "Color": (torch.linspace(0.0, 0.99, num_bins), True),
        "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
        "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
    }
    affine_ops = [
        "Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"
    ]
    return aug_space, affine_ops
