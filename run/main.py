from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Train the model using your custom dataset for 50 epochs on GPU 0
    results = model.train(data="my_data.yaml", epochs=50, device='0', batch = 2, imgsz = 416)  # change to your dataset path    