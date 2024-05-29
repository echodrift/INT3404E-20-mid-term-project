"""This module contains augmentation classes"""
from enum import Enum

import albumentations as A
import cv2
import numpy as np
from typing import List, Tuple

"""
box: x_center, y_center, width, height
x_center = x_center_pixel / image_width
y_center = y_center_pixel / image_height
width = box_width / image_width
height = box_height / image_height
"""


class CropAugmentation:
    """
    Cropping image for data augmentation
    """

    def __init__(self, images: List[np.ndarray], bboxes_list: List[List[float]]):
        """
        Constructor for CropAugmentation

        Args:
            images (List[np.ndarray]): List of image for cropping
            bboxes_list (List[List[float]]): List of list of image bounding boxes
        """
        # Crop all images
        transformed_list = [
            self._crop(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        # Save the cropped images and bounding boxes
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        """Getitem for CropAugmentation

        Args:
            idx (int): Index of image

        Returns:
            tuple: Tuple of image and list of bounding boxes
        """ 
        return self.images[idx], self.bboxes_list[idx]

    def _crop(self, image: np.ndarray, bboxes: List[float]) -> Tuple[np.ndarray, List[float]]:
        """
        Crop the image and its bounding boxes using the albumentation library

        Args:
            image (np.ndarray): The image
            bboxes (List[float]): The bounding boxes of the image

        Returns:
            tuple: Tuple of the cropped image and its bounding boxes
        """
        # Create an albumentation transform that crops the image
        transform = A.Compose(
            [
                # Crop the image randomly
                A.RandomCrop(width=500, height=500),
                # Randomly adjust the brightness and contrast of the image
                A.RandomBrightnessContrast(p=0.1),
            ],
            # Set the bounding box parameters
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        )

        # Apply the transform to the image and its bounding boxes
        transformed = transform(image=image, bboxes=bboxes)
        # Get the cropped image and its bounding boxes
        transformed_image: np.ndarray = transformed["image"]
        transformed_bboxes: List[float] = transformed["bboxes"]
        return transformed_image, transformed_bboxes


class FlipMode(Enum):
    horizontal = "horizontal"
    vertical = "vertical"


class FlipAugmentation:
    """
    Flipping image for data augmentation
    Two mode: horizontal flip and vertical flip
    """

    def __init__(self, images: List[np.ndarray], bboxes_list: List[List[float]], mode: str = "horizontal"):
        """
        Constructor for FlipAugmentation

        Args:
            images (List[np.ndarray]): List of images for flipping
            bboxes_list (List[List[float]]): List of list of bounding boxes
            mode (str): The flip mode
        """
        self.mode = mode

        transformed_list = (
            [
                # Flip the image horizontally
                self._flip_horizontal(image, bboxes)
                for image, bboxes in zip(images, bboxes_list)
            ]
            if mode == "horizontal"
            else [
                # Flip the image vertically
                self._flip_vertical(image, bboxes)
                for image, bboxes in zip(images, bboxes_list)
            ]
        )
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        """
        Getitem for FlipAugmentation
        Get the image and its bounding boxes using the index

        Args:
            idx (int): Index of image

        Returns:
            tuple: Tuple of the flipped image and its bounding boxes
        """
        return self.images[idx], self.bboxes_list[idx]

    def _flip_horizontal(self, image: np.ndarray, boxes: List[float]) -> Tuple[np.ndarray, List[float]]:
        """
        Flip the image horizontally and its bounding boxes

        Args:
            image (np.ndarray): The image
            boxes (List[float]): The bounding boxes of the image

        Returns:
            tuple: Tuple of the flipped image and its bounding boxes
        """
        # transform = A.Compose(
        #     [A.HorizontalFlip(p=1)],
        #     bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        # )

        # transformed = transform(image=image, bboxes=boxes)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        # return transformed_image, transformed_bboxes

        def get_flip(num: float) -> float:
            """
            Flip the value of a number

            This function takes a number as input and flips it.
            The flipped value is calculated by subtracting the number from 1.

            Args:
                num (float): The number to flip

            Returns:
                float: The flipped value of the number
            """
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

    def _flip_vertical(self, image: np.ndarray, boxes: List[float]) -> Tuple[np.ndarray, List[float]]:
        """
        Flip the image vertically and its bounding boxes

        Args:
            image (np.ndarray): The image
            boxes (List[float]): The bounding boxes of the image

        Returns:
            tuple: Tuple of the flipped image and its bounding boxes
        """

        # transform = A.Compose(
        #     [A.VerticalFlip(p=1)],
        #     bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        # )

        # transformed = transform(image=image, bboxes=boxes)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        # return transformed_image, transformed_bboxes

        def get_flip(num: float) -> float:
            """
            Flip the value of a number

            This function takes a number as input and flips it.
            The flipped value is calculated by subtracting the number from 1.

            Args:
                num (float): The number to flip

            Returns:
                float: The flipped value of the number
            """
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

    def __call__(self, input: np.ndarray, boxes: List[float]) -> Tuple[np.ndarray, List[float]]:
        """
        Flip the image and its bounding boxes

        Args:
            input (np.ndarray): The image
            boxes (List[float]): The bounding boxes of the image

        Returns:
            tuple: Tuple of the flipped image and its bounding boxes
        """
        # Check the mode of the flip
        if self.mode == FlipMode.horizontal:
            # Flip the image horizontally
            return self._flip_horizontal(input, boxes)
        elif self.mode == FlipMode.vertical:
            # Flip the image vertically
            return self._flip_vertical(input, boxes)
        else:
            # Raise an exception if the mode is not valid
            raise Exception(
                "There are only 'horizontal' and 'vertical' mode in FlipAugmentation"
            )


class KernelFilterAugmentation:
    """
    Apply the Gaussian blur to the image
    """

    def __init__(self, images: List[np.ndarray], bboxes_list: List[List[float]], kernel_size: int = 5):
        """
        Constructor for KernelFilterAugmentation

        Args:
            images (List[np.ndarray]): List of images for blurring
            bboxes_list (List[List[float]]): List of list of bounding boxes
            kernel_size (int): The kernel size for blurring
        """
        self.kernel_size = kernel_size
        transformed_list = [
            self._blur(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        """
        Get the image and its bounding boxes using index

        Args:
            idx (int): Index of the image

        Returns:
            tuple: Tuple of the image and its bounding boxes
        """
        return self.images[idx], self.bboxes_list[idx]

    def _blur(self, image: np.ndarray, bboxes: List[float]) -> Tuple[np.ndarray, List[float]]:
        """
        Apply the Gaussian blur to the image

        Args:
            image (np.ndarray): The image
            bboxes (List[float]): The bboxes of the image

        Returns:
            tuple: Tuple of the blurred image and the bboxes
        """
        # Create a Gaussian kernel for blurring
        kernel = cv2.getGaussianKernel(self.kernel_size, 0)
        gaussian_kernel = np.matmul(kernel, kernel.transpose())

        # Apply the Gaussian blur to the image
        blur_image = cv2.filter2D(image, -1, gaussian_kernel)
        return blur_image, bboxes


class NoiseInjectAugmentation:
    """
    Randomly injecting noise into image
    """

    def __init__(self, images: List[np.ndarray], bboxes_list: List[List[float]], mean: float = 0, std: float = 25):
        """
        Constructor for NoiseInjectAugmentation

        Args:
            images (List[np.ndarray]): List of images for adding noise
            bboxes_list (List[List[float]]): List of list of image bounding boxes
            mean (float): Mean value of the noise distribution
            std (float): Standard deviation of the noise distribution
        """
        self.mean = mean
        self.std = std
        transformed_list = [
            self._add_gaussian_noise(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        """
        Get the image and bboxes by index

        Args:
            idx (int): Index of the image

        Returns:
            tuple: Tuple of the image and bboxes
        """
        return self.images[idx], self.bboxes_list[idx]

    def _add_gaussian_noise(self, image: np.ndarray, bboxes: List[float]) -> Tuple[np.ndarray, List[float]]:
        """
        Adding gauss distribution noise into image

        Args:
            image (np.ndarray): original image
            mean (float): mean value of gauss distribution (=0 by default)
            std (float): standard deviation of gauss distribution (=25 by default)

        Returns:
            tuple: Tuple of the image after adding gauss distribution noise and its bounding boxes
        """

        print("Adding noise into image")
        print(f"Mean: {self.mean}, Standard deviation: {self.std}")
        print(f"Image shape: {image.shape}")

        # transform = A.Compose(
        #     [A.GaussNoise(var_limit=(0, self.std), mean=self.mean, always_apply=True)],
        #     bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        # )
        # transformed = transform(image=image, bboxes=bboxes)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        # return transformed_image, transformed_bboxes

        noise = np.random.normal(self.mean, self.std, image.shape)
        print(f"Noise shape: {noise.shape}")
        print(f"Noise min: {np.min(noise)}, noise max: {np.max(noise)}")
        # avoid out-of-range value and capture value from float to int
        added_noise_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        print(f"Added noise image shape: {added_noise_image.shape}")
        print(f"Added noise image min: {np.min(added_noise_image)}, added noise image max: {np.max(added_noise_image)}")
        return added_noise_image, bboxes


class RandomErasingAugmentation:
    """
    Randomly erase rectangular regions from given images.
    """
    def __init__(self, images, bboxes_list):
        """
        Initializes the RandomErasingAugmentation class with a list of images and a list of bounding box lists.
        
        Parameters:
            images (List[np.ndarray]): A list of images to be augmented.
            bboxes_list (List[List[float]]): A list of lists containing bounding box coordinates.
        
        Returns:
            None
        """
        transformed_list = [
            self._random_erase(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        """
        Get the image and its bounding boxes using index
        
        Args:
            idx (int): Index of the image
        
        Returns:
            tuple: Tuple of the image and its bounding boxes
        """
        return self.images[idx], self.bboxes_list[idx]

    def _random_erase(self, image: np.ndarray, bboxes: List[float]) -> Tuple[np.ndarray, List[float]]:
        """
        Apply random erasing to the image and its bounding boxes

        Args:
            image (np.ndarray): The image
            bboxes (List[float]): The bounding boxes of the image

        Returns:
            tuple: Tuple of the randomly erased image and its bounding boxes
        """
        # transform = torch_transforms.Compose(
        #     [
        #         torch_transforms.ToTensor(),
        #         torch_transforms.RandomErasing(
        #             p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
        #         ),
        #     ]
        # )
        # This code is replaced by the following code using the albumentation library
        transform = A.Compose(
            [A.CoarseDropout(always_apply=True)],
            # bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        return transformed_image, bboxes


class MorphologicalAugmentation:
    """
    Apply morphological operations to the images and their bounding boxes
    """

    def __init__(self, images, bboxes_list, mode: str="erosion"):
        """
        Constructor for MorphologicalAugmentation

        Args:
            images (List[np.ndarray]): List of images
            bboxes_list (List[List[float]]): List of bounding boxes
            mode (str): The morphological operation to apply
        """
        self.mode = mode 

        self.images = [self._erosion(image) if self.mode == "erosion" else self._dilation(image) for image in images]
        self.bboxes_list = bboxes_list
    def __getitem__(self, idx: int): 
        """
        Get the image and its bounding boxes using index
        
        Args:
            idx (int): Index of the image
        
        Returns:
            tuple: Tuple of the image and its bounding boxes
        """
        return self.images[idx], self.bboxes_list[idx]

    def _erosion(self, image):
        """
        Apply erosion to the image

        Args:
            image (np.ndarray): The image

        Returns:
            np.ndarray: The image after erosion
        """
        transform = A.Compose(
            [A.Morphological(operation="erosion", always_apply=True)],
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        return transformed_image
    
    def _dilation(self, image):
        """
        Apply dilation to the image

        Args:
            image (np.ndarray): The image

        Returns:
            np.ndarray: The image after dilation
        """
        transform = A.Compose(
            [A.Morphological(operation="dilation", always_apply=True)],
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        return transformed_image
