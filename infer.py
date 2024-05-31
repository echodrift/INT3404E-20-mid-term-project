# run this file to inference from the model

from ultralytics import YOLO
import os
import glob

# best yolov8 weight
weight_path = "train2/weights/best.pt"
model = YOLO(weight_path)
datasets = "training.yaml"

# specify dirs
image_dir = "datasets/images/test"
output_dir = "datasets/labels/test"

# cleanup output dir
cleanup_files = glob(output_dir + '/*')
for f in cleanup_files:
    os.remove(f)

imgs = glob(image_dir + '/*')

# infer
predictions = model.predict(source = imgs, imgsz = 928, device = [0, 1], save = False, stream = True)

index = 0
for pred in predictions:
    boxes = pred.boxes.xywhn
    labels = pred.boxes.cls
    scores = pred.boxes.conf

    # get file name
    file_base_name = os.path.basename(imgs[index]).split('.jpg')[0]
    output_file = os.path.join(output_dir, file_base_name + '.txt')

    print(f'index: {index}, output to: {output_file}')

    with open(output_file, 'w') as f:
        for box, label, score in zip(boxes, labels, scores):
            x, y, w, h = box
            f.write(f'{label} {x} {y} {w} {h} {score}\n')

        f.close()

    index += 1