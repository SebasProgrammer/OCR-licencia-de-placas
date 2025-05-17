import cv2

cap = cv2.VideoCapture('./sample4.mp4')

if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
