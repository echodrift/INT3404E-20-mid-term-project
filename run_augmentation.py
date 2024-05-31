# import os
# from glob import glob

# from utils.augmentation.augmentation import (
#     CropAugmentation,
#     FlipAugmentation,
#     KernelFilterAugmentation,
#     NoiseInjectAugmentation,
#     RandomErasingAugmentation,
# )
# from utils.dataset import CustomDataset

# BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# def augment_data():
#     train_image_url_list = glob(f"{BASE_DIR}/datasets/images/train/*.jpg")

#     train_label_url_list = [
#         url.replace("images", "labels").replace(".jpg", ".txt")
#         for url in train_image_url_list
#     ]

#     train_dataset = CustomDataset(train_image_url_list, train_image_url_list)
#     crop = CropAugmentation(train_dataset)
#     print(crop[0])
