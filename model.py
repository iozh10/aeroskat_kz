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

frame_count = 0
log_every_n_frames = 60

print("Запуск детекции. Нажмите 'q' для выхода.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.predict(
        source=frame,
        conf=0.3,
        imgsz=640,
        device=device,
        verbose=False,
        stream=False
    )

    annotated_frame = results[0].plot()

    if frame_count % log_every_n_frames == 0:
        num_objects = len(results[0].boxes)
        print(f"Кадр {frame_count}: Найдено объектов: {num_objects}")

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()