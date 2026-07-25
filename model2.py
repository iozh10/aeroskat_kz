import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
import torch

# 1. Загрузка модели YOLO
model = YOLO('last.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model.model.half()

# 2. Настройки видео
WIDTH = 640
HEIGHT = 640
FPS = 15.0
OUTPUT_FILE = 'output.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

# 3. Запускаем rpicam-vid в фоновом режиме в формате MJPEG (один раз)
cmd = [
    'rpicam-vid',
    '--width', str(WIDTH),
    '--height', str(HEIGHT),
    '--framerate', str(int(FPS)),
    '--codec', 'mjpeg',
    '--timeout', '0',       # Бесконечный поток
    '--nopreview',
    '-o', '-'              # Вывод потока в stdout
]

pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)

print(f"Запуск видеопотока. Файл сохранится в {OUTPUT_FILE}")
print("Нажмите Ctrl+C для завершения...\n")

bytes_data = b''

try:
    while True:
        # Считываем данные из потока порциями
        chunk = pipe.stdout.read(4096)
        if not chunk:
            break
        bytes_data += chunk

        # Ищем границы JPEG-кадра в байтовом потоке (Start of Image и End of Image)
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]  # Очищаем буфер

            # Декодируем кадр для OpenCV
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
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

            # Рисуем рамки и сохраняем
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

except KeyboardInterrupt:
    print("\nОстановка записи...")

finally:
    # Закрываем поток и видеофайл
    pipe.terminate()
    out.release()
    print(f"Видео успешно сохранено в '{OUTPUT_FILE}'!")