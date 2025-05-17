import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
from cvzone.Utils import cornerRect, putTextRect

ocr = PaddleOCR(use_angle_cls=True, lang="en")
license_plate_detector = YOLO('l.pt')

csv_filename = 'plates.csv'

if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_number', 'license_plate'])

cap = cv2.VideoCapture('./4k.mp4')

frame_number = 0

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

    frame_number += 1

    roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
    license_plates = license_plate_detector(roi_frame, conf=0.50)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        x1 += x_roi
        x2 += x_roi
        y1 += y_roi
        y2 += y_roi

        plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        plate_crop = cv2.resize(plate_crop, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

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

        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_number, license_text])

        cornerRect(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), l=10, rt=2, colorR=(255, 0, 0))
        putTextRect(frame, license_text, (int(x1), int(y1) - 10), scale=3.3, thickness=2, colorR=(0, 0, 0), colorB=(255, 255, 255), border=3)

    cv2.imshow('Detección de matrículas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()