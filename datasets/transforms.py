import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as t
import torchvision.transforms.functional as tf
from torch import Tensor
from torchvision.transforms import InterpolationMode

from datasets.auto_aug import _apply_op, _augmentation_space_rand_aug, _augmentation_space_tri_aug

_FILL = tuple([int(v * 255) for v in (0.485, 0.456, 0.406)])


class Compose:
    """
    Sequential transforms for both image & mask :)
    """

    def __init__(self, transforms: List):
        """
        :param transforms: List of transforms
        """
        self.transforms = transforms

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask


class ToTensor:
    """
    Convert PIL to Tensor :)
    """

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        img = tf.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return img, mask


class Normalize:
    """
    Normalize image by mean & std (image - mean) / std :)
    """

    def __init__(self, mean, std):
        """
        :param mean: mean of each channel
        :param std: std of each channel
        """
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        img = tf.normalize(img, mean=self.mean, std=self.std)
        return img, mask


class RandomResize:
    """
    Resize image randomly :)
    """

    def __init__(self, min_max_size=Tuple[int]):
        self.min_max_size = min_max_size

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        min_s, max_s = self.min_max_size
        size = random.randint(min_s, max_s)
        img = tf.resize(img, [size, size],
                        interpolation=tf.InterpolationMode.BILINEAR)
        mask = tf.resize(mask, [size, size],
                         interpolation=tf.InterpolationMode.NEAREST)
        return img, mask


class Resize:
    """
    Resize :)
    """

    def __init__(self, size: Optional[List[int]] = None):
        if size is None: size = [1024, 1024]
        self.size = size

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        img = tf.resize(img, self.size,
                        interpolation=tf.InterpolationMode.BILINEAR)
        mask = tf.resize(mask, self.size,
                         interpolation=tf.InterpolationMode.NEAREST)
        return img, mask


class RandomCrop:
    """
    Randomly crop image :)
    """

    def __init__(self, crop_size: Optional[List[int]] = None, ignore_label: int = 255):
        if crop_size is None: crop_size = [512, 1024]
        self.ignore_label = ignore_label
        self.crop_size = crop_size

    def _pad_if_needed(self, img, padvalue):
        h, w = img.size[-2], img.size[-1]
        pad_h = max(self.crop_size[0] - h, 0)
        pad_w = max(self.crop_size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top = random.randint(0, pad_h)
            pad_bottom = pad_h - pad_top
            pad_left = random.randint(0, pad_w)
            pad_right = pad_w - pad_left
            img = tf.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=padvalue)
        return img

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        img = self._pad_if_needed(img, 0.0)
        mask = self._pad_if_needed(mask, self.ignore_label)
        crop_params = t.RandomCrop.get_params(img, (self.crop_size[0], self.crop_size[1]))
        img = tf.crop(img, *crop_params)
        mask = tf.crop(mask, *crop_params)
        return img, mask


class RandomHorizontalFlip:
    """
    Flip image horizontally :)
    """

    def __init__(self, p: float = 0.5):
        """
        :param p: probability
        """
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        if random.random() < self.p:
            img = tf.hflip(img)
            mask = tf.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    """
        Flip image vertically :)
    """

    def __init__(self, p: float = 0.5):
        """
        :param p: probability
        """
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        if random.random() < self.p:
            img = tf.vflip(img)
            mask = tf.vflip(mask)
        return img, mask


class ColorJitter:
    """
    Change brightness & contrast & saturation & hue of image :)
    """

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0, hue: float = 0.0):
        """
        :param brightness: How much to jitter brightness
        :param contrast: How much to jitter contrast.
        :param saturation: How much to jitter saturation.
        :param hue: How much to jitter hue.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.jitter = t.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        return self.jitter(img), mask


class AddNoise:
    """
    Randomly add noise to image :)
    """

    def __init__(self, factor: int = 15, p: float = 0.5):
        """
        :param factor: gaussian factor
        :param p: probability
        """
        self.p = p
        self.factor = factor

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        if random.random() < self.p:
            factor = random.uniform(0, self.factor)
            img = np.array(img)
            assert (img.dtype == np.uint8)
            gauss = np.array(np.random.normal(0, factor, img.shape))
            img = (img + gauss).clip(0, 255).astype(np.uint8)
        return img, mask


class RandomRotation:
    """
    Randomly rotate image :)
    """

    def __init__(self, degrees=10.0, p=0.2, seg_fill=255, expand=False):
        """
        :param degrees: degree rotate
        :param p: probability
        :param seg_fill: mask fill value
        :param expand: expand
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            img = tf.rotate(img, random_angle, tf.InterpolationMode.BILINEAR, self.expand, fill=[0.0, 0.0, 0.0])
            mask = tf.rotate(mask, random_angle, tf.InterpolationMode.NEAREST, self.expand, fill=[self.seg_fill, ])
        return img, mask


class RandomGrayscale:
    """
        Randomly change rgb 2 gray :)
    """

    def __init__(self, p=0.5) -> None:
        """
        :param p: probability
        """
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        if random.random() < self.p:
            img = tf.rgb_to_grayscale(img, 3)
        return img, mask


class Posterize:
    """
        Randomly posterize image :)
    """

    def __init__(self, bits: int = 2) -> None:
        self.bits = bits  # 0-8

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        return tf.posterize(img, self.bits), img


class RandAugment:
    """
    RandAugment data augmentation method based on
    "https://github.com/pytorch/vision/blob/main/torchvision/transforms/autoaugment.py" :)
    """

    def __init__(
            self,
            num_ops: int = 2,
            magnitude: int = 9,
            num_magnitude_bins: int = 31,
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            fill: Optional[List[int]] = None,
            ignore_value: int = 255
    ) -> None:
        """
        :param num_ops: Number of augmentation transformations to apply sequentially.
        :param magnitude: Magnitude for all the transformations.
        :param num_magnitude_bins: The number of different magnitude values.
        :param interpolation: Desired interpolation enum
        :param fill: Pixel fill value for the area outside the transformed in image
        :param ignore_value:Pixel fill value for the area outside the transformed in mask
        """
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.mask_interpolation = InterpolationMode.NEAREST
        self.fill = fill if fill is not None else _FILL
        self.fill_mask = ignore_value

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        fill = self.fill
        channels, height, width = tf.get_dimensions(img)

        op_meta, affine_ops = _augmentation_space_rand_aug(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            if op_name in affine_ops:
                mask = _apply_op(mask, op_name, magnitude, interpolation=self.mask_interpolation, fill=self.fill_mask)
        return img, mask


class TrivialAugmentWide:
    """
        Dataset-independent data-augmentation with TrivialAugment Wide based on
        "https://github.com/pytorch/vision/blob/main/torchvision/transforms/autoaugment.py" :)
        """

    def __init__(
            self,
            num_magnitude_bins: int = 31,
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            fill: Optional[List[int]] = None,
            ignore_value: int = 255
    ) -> None:
        """
        :param num_magnitude_bins: The number of different magnitude values.
        :param interpolation: Desired interpolation enum
        :param fill: Pixel fill value for the area outside the transformed in image
        :param ignore_value:Pixel fill value for the area outside the transformed in mask
        """
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.interpolation = interpolation
        self.mask_interpolation = InterpolationMode.NEAREST
        self.fill = fill if fill is not None else _FILL
        self.fill_mask = ignore_value

    def __call__(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): img to be transformed.
            mask (PIL Image or Tensor): Mask to be transformed.
        Returns:
            PIL img or Tensor: Transformed img.
        """
        fill = self.fill
        op_meta, affine_ops = _augmentation_space_tri_aug(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        if op_name in affine_ops:
            mask = _apply_op(mask, op_name, magnitude, interpolation=self.mask_interpolation, fill=self.fill_mask)
        return img, mask


def get_augs(cfgs) -> Tuple[Compose, Compose]:
    """
    get augmentations
    :param cfgs config variables
    """
    train_transforms = []
    for aug in cfgs.TRAIN_AUGS:
        if aug == "ToTensor":
            train_transforms.append(ToTensor())
        elif aug == "Normalize":
            train_transforms.append(Normalize(mean=cfgs.MEAN, std=cfgs.STD))
        elif aug == "RandomResize":
            train_transforms.append(RandomResize(min_max_size=cfgs.MIN_MAX_SIZE))
        elif aug == "Resize":
            train_transforms.append(Resize(size=cfgs.TRAIN_SIZE))
        elif aug == "RandomCrop":
            train_transforms.append(RandomCrop(crop_size=cfgs.TRAIN_SIZE, ignore_label=cfgs.IGNORE_LABEL))
        elif aug == "RandomHorizontalFlip":
            train_transforms.append(RandomHorizontalFlip())
        elif aug == "RandomVerticalFlip":
            train_transforms.append(RandomVerticalFlip())
        elif aug == "ColorJitter":
            train_transforms.append(ColorJitter(brightness=cfgs.CJ_BRIGHTNESS, contrast=cfgs.CJ_CONTRAST,
                                                saturation=cfgs.CJ_SATURATION, hue=cfgs.CJ_HUE))
        elif aug == "RandomRotation":
            train_transforms.append(RandomRotation(degrees=cfgs.ROTATION_DEGREE, seg_fill=cfgs.IGNORE_LABEL))
        elif aug == "RandomGrayscale":
            train_transforms.append(RandomGrayscale())
        elif aug == "Posterize":
            train_transforms.append(Posterize(bits=cfgs.POSTERIZE_BITS))
        elif aug == "RandAugment":
            train_transforms.append(RandAugment(num_ops=cfgs.RANDAUG_NUM_OPS, magnitude=cfgs.RANDAUG_MAG,
                                                num_magnitude_bins=cfgs.RANDAUG_NUM_MAG_BINS,
                                                ignore_value=cfgs.IGNORE_LABEL))
        elif aug == "TrivialAugmentWide":
            train_transforms.append(TrivialAugmentWide(num_magnitude_bins=cfgs.TRIVIAL_NUM_MAG_BINS,
                                                       ignore_value=cfgs.IGNORE_LABEL))
        else:
            raise NotImplemented

    valid_transforms = [
        Resize(size=cfgs.VAL_SIZE),
        ToTensor(),
        Normalize(mean=cfgs.MEAN, std=cfgs.STD)
    ]

    return Compose(train_transforms), Compose(valid_transforms)
