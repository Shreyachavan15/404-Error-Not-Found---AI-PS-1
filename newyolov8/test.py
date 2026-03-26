import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/plastic_detector_auto11/weights/best.pt")

cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False, conf=0.5)  # 50% confidence
    annotated_frame = results[0].plot()

    cv2.imshow("Plastic Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()