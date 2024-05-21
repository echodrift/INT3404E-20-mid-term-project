import cv2 as cv
import numpy as np
import albumentations as A


class KernelFilterAugmentation:
    def __init__(self, images, bboxes_list, kernel_size=5):
        self.kernel_size = kernel_size
        transformed_list = [
            self._blur(image, bboxes) for image, bboxes in zip(images, bboxes_list)
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
