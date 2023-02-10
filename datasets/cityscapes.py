import os

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from typing import Dict
from torch import Tensor

from configs.cityscapes import LABEL_MAPPING, CLASSES, PHASES
from datasets.base import BaseDataset


def convert_labels(label: np.ndarray) -> np.ndarray:
    """
    convert all classes of cityscapes to 19 classes :)
    :param label: old label
    :return: new label
    """
    temp = label.copy()
    for k, v in LABEL_MAPPING.items():
        label[temp == k] = v
    return label


def show_image_city(image: PIL.Image.Image) -> None:
    """
    plot image :)
    :param image: image
    """
    plt.clf()
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def show_numpy_mask_city(mask: PIL.Image.Image) -> None:
    """
    plot mask + guide :)
    :param mask:
    """
    colors = []
    labels = []
    for v in CLASSES.values():
        colors.append(list(v["color"]))
        labels.append(v["name"])
    colors = np.array(colors, dtype=np.uint8)
    handles = [Rectangle((0, 0), 1, 1, color=_c / 255) for _c in colors]
    mask.putpalette(colors)
    plt.imshow(mask)
    plt.axis('off')
    plt.show()

    plt.imshow(np.ones((1, 1, 3)))
    plt.legend(handles, labels, loc="center")
    plt.axis('off')
    plt.show()


class CityScapes(BaseDataset):
    """
    CityScapes Dataset Class :)
    """
    def __init__(self, phase, root, transforms=None, debug: bool = False) -> None:
        """
        :param phase: train or validation
        :param root: dataset directory
        :param transforms: data augmentations for change datas
        :param debug: watch image & mask
        """
        super().__init__()
        assert phase in PHASES, f"{phase} not in {PHASES} :)"
        if not os.path.isfile("./train_val_paths.json"):
            self.create_json_paths(root, PHASES)
        self.files = self.read_json_file(phase)
        self.debug = debug
        self.transforms = transforms

    def __getitem__(self, idx) -> Dict[Tensor, Tensor]:
        image, mask = self.files[idx]["Image"], self.files[idx]["Mask"]
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = convert_labels(mask)
        image = PIL.Image.fromarray(image)
        mask = PIL.Image.fromarray(mask)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        if self.debug:
            show_image_city(image)
            show_numpy_mask_city(mask)
        return image, mask

    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    ds = CityScapes(phase="val", root="../data/Cityscapes/", debug=True)
    image, mask = ds[0]