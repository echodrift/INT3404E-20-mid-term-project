import albumentations as A


class MorphologicalAugmentation:
    def __init__(self, images, bboxes_list, mode: str = "erosion"):
        self.mode = mode

        self.images = [
            self._erosion(image) if self.mode == "erosion" else self._dilation(image)
            for image in images
        ]
        self.bboxes_list = bboxes_list

    def __getitem__(self, idx: int):
        return self.images[idx], self.bboxes_list[idx]

    def _erosion(self, image):
        transform = A.Compose(
            [A.Morphological(operation="erosion", always_apply=True)],
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        return transformed_image

    def _dilation(self, image):
        transform = A.Compose(
            [A.Morphological(operation="dilation", always_apply=True)],
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        return transformed_image
