from glob import glob

from utils import CustomDataset

train_image_url_list = glob(
    "/home/lvdthieu/Documents/Projects/ip-mid-term/datasets/images/train/*.jpg"
)

train_label_url_list = [
    url.replace("images", "labels").replace(".jpg", ".txt")
    for url in train_image_url_list
]

train_dataset = CustomDataset(train_image_url_list, train_label_url_list)
print(CustomDataset.get_size(train_dataset[1]))
