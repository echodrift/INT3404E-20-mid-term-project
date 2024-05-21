import cv2
import albumentations as A
from enum import Enum
import numpy as np


class FlipMode(Enum):
    horizontal = "horizontal"
    vertical = "vertical"


class FlipAugmentation:
    """
    Flipping image for data augmentation
    Two mode: horizontal flip and vertical flip
    """

    def __init__(self, images, bboxes_list, mode: str = "horizontal"):
        self.mode = mode

        transformed_list = (
            [
                self._flip_horizontal(image, bboxes)
                for image, bboxes in zip(images, bboxes_list)
            ]
            if mode == "horizontal"
            else [
                self._flip_vertical(image, bboxes)
                for image, bboxes in zip(images, bboxes_list)
            ]
        )
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        return self.images[idx], self.bboxes_list[idx]

    def _flip_horizontal(self, image, boxes):
        """
        box: x_center, y_center, width, height\n
        x_center = x_center_pixel / image_width\n
        y_center = y_center_pixel / image_height\n
        width = box_width / image_width\n
        height = box_height / image_height
        """

        # transform = A.Compose(
        #     [A.HorizontalFlip(p=1)],
        #     bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        # )

        # transformed = transform(image=image, bboxes=boxes)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        # return transformed_image, transformed_bboxes

        def get_flip(num):
            return 1 - num

        new_boxes = []
        for box in boxes:
            new_box = [
                get_flip(box[0]),
                box[1],
                box[2],
                box[3],
            ]
            new_boxes.append(new_box)

        return np.fliplr(image), new_boxes

    def _flip_vertical(self, image, boxes):
        """
        box: x_center, y_center, width, height\n
        x_center = x_center_pixel / image_width\n
        y_center = y_center_pixel / image_height\n
        width = box_width / image_width\n
        height = box_height / image_height
        """

        # transform = A.Compose(
        #     [A.VerticalFlip(p=1)],
        #     bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        # )

        # transformed = transform(image=image, bboxes=boxes)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        # return transformed_image, transformed_bboxes

        def get_flip(num):
            return 1 - num

        new_boxes = []
        for box in boxes:
            new_box = [
                box[0],
                get_flip(box[1]),
                box[2],
                box[3],
            ]
            new_boxes.append(new_box)
        return np.flipud(image), new_boxes

    def __call__(self, input, boxes):
        if self.mode == FlipMode.horizontal:
            return self._flip_horizontal(input, boxes)
        elif self.mode == FlipMode.vertical:
            return self._flip_vertical(input, boxes)
        else:
            raise Exception(
                "There are only 'horizontal' and 'vertical' mode in FlipAugmentation"
            )
