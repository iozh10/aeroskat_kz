import cv2
import numpy as np
import subprocess
import time
from ultralytics import YOLO
import torch

model = YOLO('last.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model.model.half()

WIDTH = 640
HEIGHT = 640
FPS = 10.0
OUTPUT_FILE = 'output.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

print(f"Запуск обработки. Итоговое видео будет сохранено в файл: {OUTPUT_FILE}")
print("Для остановки нажмите Ctrl+C в терминале.\n")

try:
    while True:
        result = subprocess.run([
            'rpicam-still',
            '--width', str(WIDTH),
            '--height', str(HEIGHT),
            '--quality', '80',
            '--timeout', '1',
            '--encoding', 'jpg',
            '--nopreview',
            '-o', '-'
        ], capture_output=True, timeout=2)

        if result.returncode != 0 or not result.stdout:
            print("Ошибка захвата кадра с камеры. Пропускаем...")
            continue

        img_np = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        results = model.predict(
            source=frame,
            conf=0.3,
            imgsz=WIDTH,
            device=device,
            verbose=False
        )

        boxes = results[0].boxes
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                print(f"Обнаружено: {class_name:<10} | Точность: {confidence*100:.1f}% | Координаты: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

except KeyboardInterrupt:
    print("\nОстановка записи пользователем (Ctrl+C).")

finally:
    out.release()
    print(f"Видео успешно сохранено! Вы можете найти файл '{OUTPUT_FILE}' на флешке.")