import cv2
from ultralytics import YOLO

yolo_model = YOLO('yolo11x.pt')

vehicles = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

cap = cv2.VideoCapture('./sample4.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    results = yolo_model(frame)

    for result in results:
        for box, class_id in zip(result.boxes.xyxy, result.boxes.cls):
            class_id = int(class_id)
            if class_id in vehicles:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, vehicles[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detección de Vehículos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
