import os
from glob import glob
from typing import List, NamedTuple, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from augmentation import (
    CropAugmentation,
    FlipAugmentation,
    KernelFilterAugmentation,
    MorphologicalAugmentation,
    NoiseInjectAugmentation,
    RandomErasingAugmentation,
)
from keras.preprocessing.image import load_img

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER = f"{BASE_DIR}/../../datasets"
TRAIN_IMG_BASE = f"{BASE_DIR}/../../datasets/images/train"
VAL_IMG_BASE = f"{BASE_DIR}/../../datasets/images/val"
TRAIN_LABEL_BASE = f"{BASE_DIR}/../../datasets/labels/train"
VAL_LABEL_BASE = f"{BASE_DIR}/../../datasets/labels/val"


class Location(NamedTuple):
    center_x: float  # center_x/image_width
    center_y: float  # center_y/image_height
    width: float  # width/image_width
    height: float  # height/image_height


class Point(NamedTuple):
    x: int
    y: int


def draw_bounding_box(img, bboxes):
    """
    Draw bounding boxes on the image

    Args:
        img (numpy.ndarray): The image to draw bounding boxes on
        bboxes (List[Location]): The bounding boxes to draw

    Returns:
        None:
    """
    img = np.ascontiguousarray(img)
    H, W, C = img.shape
    for box in bboxes:
        center_x = box.center_x * W
        center_y = box.center_y * H
        height = box.height * H
        width = box.width * W
        top_left = Point(int(center_x - width / 2), int(center_y - height / 2))
        bottom_right = Point(
            int(center_x + width / 2), int(center_y + height / 2)
        )
        # Draw the bounding box
        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
    plt.imshow(img)
    plt.show()


def load_bboxes(path: str):
    """
    Load bounding boxes from a file

    Args:
        path (str): The path to the file containing the bounding boxes

    Returns:
        List[List[float]]: The list of bounding boxes
    """
    with open(path) as f:
        bboxes = []
        for line in f:
            params = list(
                map(float, line.split()[1:])
            )  # params = [x, y, width, height]
            bboxes.append(params)
        return bboxes


def convert_bboxes_to_full_labels(bboxes_list):
    """
    Convert a list of bounding boxes to a list of full labels

    Args:
        bboxes_list (List[List[float]]): The list of bounding boxes

    Returns:
        List[List[float]]: The list of full labels
    """
    labels_list = []
    for bboxes in bboxes_list:
        labels = []
        for bbox in bboxes:
            # Add the class label to the end of the bounding box
            # This is necessary for the YOLO model, which requires the class
            # label to be included in the bounding box
            label = bbox + [0]
            labels.append(label)
        labels_list.append(labels)
    return labels_list


def save(data_folder, images, bboxes_list, method):
    """
    Save the images and the bounding boxes to a file

    Args:
        data_folder (str): The path to the folder in which to save the data
        images (List[np.array]): The images to save
        bboxes_list (List[List[float]]): The bounding boxes to save
        method (str): The name of the method used to generate the bounding boxes

    Returns:
        None
    """
    i = 0
    for image, bboxes in zip(images, bboxes_list):
        # id = uuid.uuid4()
        cv2.imwrite(
            filename=f"{data_folder}/images/augmented/{method}_{i}.jpg",
            img=image,
        )
        with open(f"{data_folder}/labels/augmented/{method}_{i}.txt", "w") as f:
            for box in bboxes:
                # If the bounding box is in the format [x, y, width, height]
                if len(box) == 4:
                    params = " ".join(list(map(str, box)))
                    f.write(f"0 {params}\n")
                # If the bounding box is in the format [x, y, width, height, class]
                elif len(box) == 5:
                    params = " ".join(list(map(str, box[:4])))
                    f.write(f"0 {params}\n")
        i += 1


if __name__ == "__main__":
    train_img_paths = sorted(glob(f"{TRAIN_IMG_BASE}/*.jpg"))
    val_img_paths = sorted(glob(f"{VAL_IMG_BASE}/*.jpg"))
    train_label_paths = sorted(glob(f"{TRAIN_LABEL_BASE}/*.txt"))
    val_label_paths = sorted(glob(f"{VAL_LABEL_BASE}/*.txt"))

    bboxes_list = [
        load_bboxes(path=train_label_path)
        for train_label_path in train_label_paths
    ]
    images = [
        cv2.cvtColor(cv2.imread(train_image_path), cv2.COLOR_BGR2RGB)
        for train_image_path in train_img_paths
    ]
    horizontal_augmentation = FlipAugmentation(images, bboxes_list)
    save(
        DATA_FOLDER,
        horizontal_augmentation.images,
        horizontal_augmentation.bboxes_list,
        method="horizontal",
    )
    vertical_augmentation = FlipAugmentation(
        images, bboxes_list, mode="vertical"
    )
    save(
        DATA_FOLDER,
        vertical_augmentation.images,
        vertical_augmentation.bboxes_list,
        method="vertical",
    )
    crop_augmentation = CropAugmentation(
        images, convert_bboxes_to_full_labels(bboxes_list)
    )
    save(
        DATA_FOLDER,
        crop_augmentation.images,
        crop_augmentation.bboxes_list,
        method="crop",
    )
    kernel_augmentation = KernelFilterAugmentation(images, bboxes_list)
    save(
        DATA_FOLDER,
        kernel_augmentation.images,
        kernel_augmentation.bboxes_list,
        method="kernel",
    )
    noise_inject_augmentation = NoiseInjectAugmentation(images, bboxes_list)
    save(
        DATA_FOLDER,
        noise_inject_augmentation.images,
        noise_inject_augmentation.bboxes_list,
        method="noise_inject",
    )
    random_erase_augmentation = RandomErasingAugmentation(images, bboxes_list)
    save(
        DATA_FOLDER,
        random_erase_augmentation.images,
        random_erase_augmentation.bboxes_list,
        method="random_erase",
    )
    erosion_augmentation = MorphologicalAugmentation(
        images=images,
        bboxes_list=convert_bboxes_to_full_labels(bboxes_list),
        mode="erosion",
    )
    save(
        DATA_FOLDER,
        erosion_augmentation.images,
        erosion_augmentation.bboxes_list,
        method="erosion",
    )
    dilation_augmentation = MorphologicalAugmentation(
        images=images,
        bboxes_list=convert_bboxes_to_full_labels(bboxes_list),
        mode="dilation",
    )
    save(
        DATA_FOLDER,
        dilation_augmentation.images,
        dilation_augmentation.bboxes_list,
        method="dilation",
    )
