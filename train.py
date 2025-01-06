from ultralytics import YOLO

def train_model():
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8x-pose.pt')  # or 'yolov8l-pose.pt' for a lighter model

    # Train the model with custom settings
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,  # reduced batch size for CPU training
        patience=20,
        save=True,
        device='cpu',  # explicitly use CPU
        workers=4,  # reduced workers for CPU
        project='padel_coach',
        name='train_1',
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=2.0,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True
    )

    return results

if __name__ == "__main__":
    train_model()