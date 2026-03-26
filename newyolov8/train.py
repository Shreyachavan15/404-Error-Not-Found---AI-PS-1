from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")   # 🔥 switch to nano (VERY important)

    model.train(
        data="C:/Users/HP/OneDrive/Desktop/newyolov8/test_yolo/dataset.yaml",
        epochs=50,

        imgsz=512,      # 🔥 reduce image size
        batch=4,        # 🔥 reduce batch

        name="plastic_detector_auto",

        workers=2,      # 🔥 lower workers
        device=0,

        amp=True,
        cache=False
    )

if __name__ == "__main__":
    main()