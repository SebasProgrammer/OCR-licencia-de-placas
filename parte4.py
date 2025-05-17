import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
from cvzone.Utils import cornerRect, putTextRect

ocr = PaddleOCR(use_angle_cls=True, lang="en")

coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('l.pt')

cap = cv2.VideoCapture('./sample4.mp4')

vehicles = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

bytetrack_tracker = "bytetrack.yaml"

vehicle_plates = {}

ret, frame = cap.read()
if not ret:
    print("Error al leer el video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

roi = cv2.selectROI("Seleccione la región de interés", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Seleccione la región de interés")

x_roi, y_roi, w_roi, h_roi = roi

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()

    results = coco_model.track(roi_frame, persist=True, tracker=bytetrack_tracker, classes=list(vehicles.keys()))

    vehicle_tracks = {}

    if results[0].boxes.id is not None:
        for box, track_id, class_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy()
            track_id = int(track_id)
            class_id = int(class_id)

            x1 += x_roi
            x2 += x_roi
            y1 += y_roi
            y2 += y_roi

            vehicle_tracks[track_id] = (x1, y1, x2, y2)

    license_plates = license_plate_detector(roi_frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        x1 += x_roi
        x2 += x_roi
        y1 += y_roi
        y2 += y_roi

        for track_id, (xcar1, ycar1, xcar2, ycar2) in vehicle_tracks.items():
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                plate_crop = cv2.resize(plate_crop, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
                plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                if score > 0.6:
                    ocr_result = ocr.ocr(plate_gray, cls=True)

                license_text = ""
                max_conf = 0
                conf = 0.0

                if ocr_result and isinstance(ocr_result, list):  
                    for line in ocr_result:
                        if isinstance(line, list):
                            for word_info in line:
                                if isinstance(word_info, list) and len(word_info) > 1:
                                    text, conf = word_info[1]
                                    if conf > 0.7:
                                        if conf > max_conf:
                                            max_conf = conf
                                            license_text = text.upper().replace(" ", "")

                if track_id not in vehicle_plates:
                    vehicle_plates[track_id] = (license_text, conf)
                elif conf > vehicle_plates[track_id][1]:
                    vehicle_plates[track_id] = (license_text, conf)

                cornerRect(frame, (int(xcar1), int(ycar1), int(xcar2 - xcar1), int(ycar2 - ycar1)), l=10, rt=2, colorR=(255, 0, 0))

                putTextRect(frame, f'Car {track_id}', (int(xcar1), int(ycar1) - 10), scale=0.8, thickness=2, colorR=(255, 0, 0), colorB=(255, 255, 255))

                putTextRect(frame, vehicle_plates[track_id][0], (int(x1), int(y1) - 10), scale=1.3, thickness=2, colorR=(0, 0, 0), colorB=(255, 255, 255), border=3)

    cv2.imshow('Vehicle and License Plate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
