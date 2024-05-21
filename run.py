from glob import glob
import os

from utils.dataset import CustomDataset

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

train_image_url_list = glob(
    f"{BASE_DIR}/datasets/images/train/*.jpg"
)

train_label_url_list = [
    url.replace("images", "labels").replace(".jpg", ".txt")
    for url in train_image_url_list
]

train_dataset = CustomDataset(train_image_url_list, train_label_url_list)
print(CustomDataset.get_size(train_dataset[1]))
CustomDataset.draw_bounding_box(train_dataset[1])
