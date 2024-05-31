# Image Processing: Sino-Nom character localization

## Introduction

This project aims to localize Sino-Nom characters in images using image processing techniques and the YOLO (You Only Look Once) algorithm.

## Description

The project is structured as follows:

- `data.yaml`: Contains the configuration for the dataset.
- `datasets`: Contains the images and labels for training and validation.
- `utils/augmentation/run.py`: The main script to augment the dataset.
- `utils`: Contains utility scripts for data augmentation and dataset handling.
- `yolo_baseline.py`: Code to run YOLO baseline, no improvement to generate a baseline to compare between approaches
- `yolov8.py`: Code to train YOLO v8 (final code of us)
- `run_v9.sh`: The script to train YOLO v9 model

## Project structure

```
.
├── datasets # put your dataset here and update data.yaml
│   ├── images
│   └── labels
├── data.yaml
├── eval.py # use to evaluate your model
├── infer.py # use to infer your model
├── notebooks
│   ├── augmentation.ipynb # run this to perform data augmentation
│   ├── nom-localization.ipynb
│   └── not-gonna-make-it.ipynb # kaggle notebook for fine-tuning yolov8 and v9
├── README.md
├── report.pdf          # full report and training plans with analysis from our group
├── requirements.txt      # dependencies needed for code in this repo
├── run_augmentation.py   # run this to perform data augmentation
├── run_v9.sh            #run this train v9 model (not included with the weight)
├── train # best checkpoint for yolov9
│   ├── args.yaml
│   └── weights
├── train2 # best checkpoint for yolov8
│   ├── args.yaml
│   └── weights
├── training.yaml # custom dataset path
├── utils # libs for data augmentation
│   ├── augmentation
│   ├── dataset.py
├── yolo_baseline.py # the baseline model for comparision
└── yolov8.py # our training plan
```

## Result

![result-table](result.png)

## Reproduce

To reproduce the results, follow these steps:

1. Clone the repository.
2. Install the necessary dependencies: `pip install -r requirements.txt`.
3. Get our data from <a href="https://www.kaggle.com/datasets/ngolehoang/sinom-augment-data">Augment-Data </a> or you can make own dataset and put data to the right folder (this will not guarantee to reproduce the same result).
4. If you want to train from scratch run `sh run_v9.sh` or else you can use our code to finetune YOLOv9 model
   1. Get a pretrained model weights from anywhere you can get from ""
   2. Run `python yolov8.py` (please read the file)
## Authors
This repository was made by my team consist of 4 members:
- [Luu Van Duc Thieu](https://github.com/echodrift)
- [Ngo Le Hoang](https://github.com/armistcxy)
- [Nguyen Thanh Phat](https://github.com/aqu4holic)
- [Le Hong Duc](https://github.com/tedomi2705)

## License
[MIT](https://choosealicense.com/licenses/mit/)
