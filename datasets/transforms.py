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


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets based on
    https://github.com/pytorch/vision/blob/main/references/classification/transforms.py :)
    """

    def __init__(self, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        """
        :param p: probability of the batch being transformed.
        :param alpha: hyperparameter of the Beta distribution used for cutmix.
        :param inplace: boolean to make this transform inplace
        """
        super().__init__()
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
            batch (Tensor): Float tensor of size (B, C, H, W)
            mask (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if mask.ndim != 4:
            raise ValueError(f"Target ndim should be 1. Got {mask.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if mask.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {mask.dtype}")

        if not self.inplace:
            batch = batch.clone()
            mask = mask.clone()

        if torch.rand(1).item() >= self.p:
            return batch, mask

        batch_rolled = batch.roll(1, 0)
        mask_rolled = mask.roll(1, 0)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = tf.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        mask[:, :, y1:y2, x1:x2] = mask_rolled[:, :, y1:y2, x1:x2]

        return batch, mask
