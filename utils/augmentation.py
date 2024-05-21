from enum import Enum

import albumentations as A
import numpy as np


class CropAugmentation:
    """
    Cropping image for data augmentation
    """

    def __init__(self, images, bboxes_list):
        transformed_list = [
            self._crop(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        return self.images[idx], self.bboxes_list[idx]

    def _crop(self, image, bboxes):
        """
        box: x_center, y_center, width, height
        x_center = x_center_pixel / image_width
        y_center = y_center_pixel / image_height
        width = box_width / image_width
        height = box_height / image_height
        """

        transform = A.Compose(
            [
                A.RandomCrop(width=500, height=500),
                A.RandomBrightnessContrast(p=0.1),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        )

        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        return transformed_image, transformed_bboxes


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


class KernelFilterAugmentation:
    def __init__(self, images, bboxes_list, kernel_size=5):
        self.kernel_size = kernel_size
        transformed_list = [
            self._blur(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        return self.images[idx], self.bboxes_list[idx]

    def _blur(self, image, bboxes):

        # transform = A.Compose(
        #     [A.GaussianBlur(blur_limit=(self.kernel_size, self.kernel_size), always_apply=True)],
        #     bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        # )
        # transformed = transform(image=image, bboxes=bboxes)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        # return transformed_image, transformed_bboxes

        kernel = cv.getGaussianKernel(self.kernel_size, 0)
        gaussian_kernel = np.matmul(kernel, kernel.transpose())

        blur_image = cv.filter2D(image, -1, gaussian_kernel)
        return blur_image, bboxes


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


class RandomErasingAugmentation:
    def __init__(self, images, bboxes_list):
        transformed_list = [
            self._random_erase(image, bboxes)
            for image, bboxes in zip(images, bboxes_list)
        ]
        self.images = [transformed[0] for transformed in transformed_list]
        self.bboxes_list = [transformed[1] for transformed in transformed_list]

    def __getitem__(self, idx: int):
        return self.images[idx], self.bboxes_list[idx]

    def _random_erase(self, image, bboxes):

        # transform = torch_transforms.Compose(
        #     [
        #         torch_transforms.ToTensor(),
        #         torch_transforms.RandomErasing(
        #             p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
        #         ),
        #     ]
        # )
        transform = A.Compose(
            [A.CoarseDropout(always_apply=True)],
            # bbox_params=A.BboxParams(format="yolo", min_visibility=0.5),
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        # transformed_bboxes = transformed["bboxes"]
        return transformed_image, bboxes

        kernel = cv.getGaussianKernel(self.kernel_size, 0)
        gaussian_kernel = np.matmul(kernel, kernel.transpose())

        blur_image = cv.filter2D(image, -1, gaussian_kernel)
        return blur_image, bboxes
