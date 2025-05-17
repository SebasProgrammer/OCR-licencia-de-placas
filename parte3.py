import cv2
import numpy as np
from ultralytics import YOLO

coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('l.pt')

cap = cv2.VideoCapture('./sample4.mp4')

vehicles = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ret, frame = cap.read()
roi = cv2.selectROI("Seleccione la región de interés", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Seleccione la región de interés")

x_roi, y_roi, w_roi, h_roi = roi

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
    results = coco_model.track(roi_frame, persist=True)

    for result in results:
        if result.boxes.id is None:
            continue

        for box, class_id, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
            class_id = int(class_id)
            track_id = int(track_id)

            if class_id in vehicles:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                x1 += x_roi
                x2 += x_roi
                y1 += y_roi
                y2 += y_roi

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {track_id} {vehicles[class_id]}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                vehicle_crop = frame[y1:y2, x1:x2].copy()
                plates_results = license_plate_detector(vehicle_crop)

                for result in plates_results:
                    for box in result.boxes.xyxy:
                        px1, py1, px2, py2 = map(int, box.cpu().numpy())
                        px1 += x1
                        px2 += x1
                        py1 += y1
                        py2 += y1

                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        cv2.putText(frame, "Placa", (px1, py1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Detección y Tracking de Vehículos y Placas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
