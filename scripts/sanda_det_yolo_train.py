from ultralytics import YOLO


if __name__ == '__main__':

    # Load a model
    model = YOLO("/Users/zhaosilei/PycharmProjects/Benz/Sanda_action_recognation/runs_1/detect/train/weights/best.pt")

    # Train the model with MPS
    results = model.train(data="sanda-det.yaml", epochs=150, imgsz=640, device="mps",
                          save=True, cos_lr=True, close_mosaic=30)
