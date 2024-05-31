from ultralytics import YOLO
import cv2
import os
from glob import glob

# best yolov8 weight
weight_path = "train2/weights/best.pt"
model = YOLO(weight_path)
datasets = "training.yaml"

results = model.val(data = datasets, imgsz = 928, batch = 8, device=[0, 1])
img = cv2.imread("datasets/images/val/nlvnpf-0137-01-045.jpg")