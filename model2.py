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
        try:
            # Захват одного кадра через rpicam-still
            result = subprocess.run([
                'rpicam-still',
                '--width', str(WIDTH),
                '--height', str(HEIGHT),
                '--quality', '80',
                '--timeout', '50',      # Уменьшили задержку камеры до 50 мс
                '--encoding', 'jpg',
                '--nopreview',
                '-o', '-'              # Вывод в stdout
            ], capture_output=True, timeout=5) # Увеличили таймаут Python до 5 секунд

            if result.returncode != 0 or not result.stdout:
                continue

            # Преобразование байтов JPEG в numpy-массив
            img_np = np.frombuffer(result.stdout, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Предикт модели YOLO
            results = model.predict(
                source=frame,
                conf=0.3,
                imgsz=WIDTH,
                device=device,
                verbose=False
            )

            # Отрисовка рамок поверх кадра
            annotated_frame = results[0].plot()

            # Запись обработанного кадра в видеофайл
            out.write(annotated_frame)

        except subprocess.TimeoutExpired:
            # Если камера "задумалась", просто пропускаем этот кадр, а не роняем скрипт
            print("Превышено время ожидания кадра от камеры, пропуск...")
            continue

except KeyboardInterrupt:
    print("\nОстановка записи пользователем (Ctrl+C).")

finally:
    # Обязательно освобождаем ресурс записи, чтобы файл не бился
    out.release()
    print(f"Видео успешно сохранено в '{OUTPUT_FILE}'.")