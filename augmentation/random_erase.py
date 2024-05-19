import cv2 as cv
import numpy as np
import albumentations as A
import torchvision.transforms as torch_transforms


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
