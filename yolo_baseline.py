from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="data.yaml", epochs=10, imgsz=(608, 900), seed=1)
    metrics = model.val()
    path = model.export(format="onnx")
    print("Metrics:", metrics, sep="\n")
    print("-" * 100)
    print("Path:", path, sep="\n")
