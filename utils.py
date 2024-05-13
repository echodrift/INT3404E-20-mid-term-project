import os
from glob import glob
from typing import List, NamedTuple, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class Location(NamedTuple):
    center_x: float  # center_x/image_width
    center_y: float  # center_y/image_height
    width: float  # width/image_width
    height: float  # height/image_height


class Point(NamedTuple):
    x: int
    y: int


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_url_list: List[str], label_url_list: List[str]):
        super(CustomDataset, self).__init__()
        self.img_url_list = img_url_list
        self.label_url_list = label_url_list

    def __len__(self):
        return len(self.img_url_list)

    def _get_box_location(self, string: str) -> Location:
        return Location(*map(float, string.split()[1:]))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Location]]:
        img_url = self.img_url_list[idx]
        label_url = self.label_url_list[idx]
        img = cv2.imread(img_url)[..., ::-1]
        with open(label_url) as f:
            bounding_boxes = list(map(self._get_box_location, f.read().splitlines()))
        return img, bounding_boxes

    @staticmethod
    def draw_bounding_box(img, bounding_boxes):
        H, W, C = img.shape
        for box in bounding_boxes:
            center_x = box.center_x * W
            center_y = box.center_y * H
            height = box.height * H
            width = box.width * W
            top_left = Point(int(center_x - width / 2), int(center_y - height / 2))
            bottom_right = Point(int(center_x + width / 2), int(center_y + height / 2))
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        plt.imshow(img)
