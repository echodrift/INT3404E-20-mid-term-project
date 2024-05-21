import numpy as np
import cv2
import albumentations as A
from enum import Enum


class NoiseInjectAugmentation:
    """
    Randomly injecting noise into image
    """

    def __init__(self, images, bboxes_list, mean=0, std=25):
        self.mean = mean
        self.std = std
        transformed_list = [
            self._add_gaussian_noise(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        return self.images[idx], self.bboxes_list[idx]

    def _add_gaussian_noise(self, image, bboxes):
        """
        Adding gauss distribution noise into image

        Params:
            image: original image
            mean: mean value of gauss distribution (=0 by default)
            std: standard deviation of gauss distribution (=25 by default)

        Returns:
            numpy.ndarray: image after adding gauss distribution noise
        """

        # transform = A.Compose(
        #     [A.GaussNoise(var_limit=(0, self.std), mean=self.mean, always_apply=True)],
        #     bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        # )
        # transformed = transform(image=image, bboxes=bboxes)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        # return transformed_image, transformed_bboxes

        noise = np.random.normal(self.mean, self.std, image.shape)
        # avoid out-of-range value and capture value from float to int
        added_noise_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return added_noise_image, bboxes

    def save(save_image_path: str, save_label_path: str) -> None:
        pass
