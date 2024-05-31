from ultralytics import YOLO

if __name__ == "__main__":
    # YOLO v8x
    model = YOLO("yolov8x.pt")
    dataset = "training.yaml"

    # the full training plan is in the report, please read it before proceeding

    # change the params below according to your hardware specs
    # but dont change these params!:
    # imgsz, seed

    # steps to reproduce

    # first 100 epochs with given dataset
    results = model.train(
        data=dataset,
        batch=4,
        epochs=100,
        imgsz=928,
        seed=18022004,
        device=[0, 1],
        save_period=5,
        name="v8x_base",
    )

    # take the `best.pt` checkpoint from `v8x_base` and train next 100 or 200 epochs with augmented data
    model = "/path/to/the/best/weight"
    dataset = "/augmented/data/set"
    results = model.train(
        data=dataset,
        batch=4,
        epochs=100,
        imgsz=928,
        seed=18022004,
        device=[0, 1],
        save_period=5,
        name="v8x_augmented",
    )
