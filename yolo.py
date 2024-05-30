from ultralytics import YOLO

if __name__ == "__main__":
    # YOLO v8x
    model = YOLO("yolov8x.pt")
    dataset = "training.yaml"
    results = model.train(data = dataset, batch = 4, epochs = 100, imgsz = 928, seed = 18022004, device = [0, 1], save_period = 5, name = 'v8x_merge_aug12')
    # YOLO v9
    model_path = '/kaggle/working/image_processing/yolov9-e-converted.pt'