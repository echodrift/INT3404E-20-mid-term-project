import cv2
import albumentations as A
from enum import Enum
import numpy as np


class CropAugmentation:
    """
    Cropping image for data augmentation
    """

    def __init__(self, images, bboxes_list):

        transformed_list = [
            self._crop(image, bboxes) for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        return self.images[idx], self.bboxes_list[idx]

    def _crop(self, image, bboxes):
        """
        box: x_center, y_center, width, height\n
        x_center = x_center_pixel / image_width\n
        y_center = y_center_pixel / image_height\n
        width = box_width / image_width\n
        height = box_height / image_height
        """

        transform = A.Compose(
            [A.RandomCrop(width=500, height=500), A.RandomBrightnessContrast(p=0.1)],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        )

        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        return transformed_image, transformed_bboxes
