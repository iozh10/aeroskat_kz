import cv2
from ultralytics import YOLO
import torch

model = YOLO('last.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model.model.half()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


print("Нажмите 'q' для выхода.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр с камеры.")
        break

    # Предикт модели
    results = model.predict(
        source=frame,
        conf=0.3,
        imgsz=640,
        device=device,
        verbose=False,
        stream=False
    )


    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Печать данных в консоль
            print(f"Буква:{class_name:<15} | Точность: {confidence*100:.1f}% | Коорд: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()