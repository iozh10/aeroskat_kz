import cv2
import numpy as np
import subprocess
import threading
import time
import urllib.request
from flask import Flask, Response
from ultralytics import YOLO
import torch

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
WIDTH = 640
HEIGHT = 640
FPS = 15.0
PORT = 8080
OUTPUT_FILE = 'output.mp4'  # Файл запишется на SD-карту

# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ FLASK (СЕРВЕР)
# ==========================================
app = Flask(__name__)

def generate_stream():
    """Захват видео с камеры через rpicam-vid в формате MJPEG"""
    cmd = [
        'rpicam-vid',
        '--width', str(WIDTH),
        '--height', str(HEIGHT),
        '--framerate', str(int(FPS)),
        '--codec', 'mjpeg',
        '--timeout', '0',       # Бесконечный поток
        '--nopreview',
        '-o', '-'              # Вывод в stdout
    ]
    
    # Запускаем фоновый процесс захвата
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)
    
    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')
    finally:
        process.terminate()

@app.route('/')
def video_feed():
    """Эндпоинт для видеопотока"""
    return Response(
        generate_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def run_flask():
    """Функция запуска веб-сервера в отдельном потоке"""
    app.run(host='0.0.0.0', port=PORT, threaded=True, use_reloader=False)

# ==========================================
# 3. ОСНОВНОЙ КОД (YOLO + ЗАПИСЬ НА SD-КАРТУ)
# ==========================================
def main():
    # Запускаем Flask сервер в фоновом режиме
    server_thread = threading.Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    print(f"[*] Веб-сервер запущен на http://0.0.0.0:{PORT}")
    print("[*] Ожидаем инициализацию камеры и сервера (2 сек)...")
    time.sleep(2)

    # Загрузка модели YOLO
    print("[*] Загрузка модели YOLO...")
    model = YOLO('last.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model.model.half()

    # Подготовка модуля записи видео в MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

    # Подключаемся к локальному MJPEG-потоку
    stream_url = f"http://127.0.0.1:{PORT}/"
    stream = urllib.request.urlopen(stream_url)

    print(f"[*] Обработка началась! Видео сохраняется в файл: {OUTPUT_FILE}")
    print("[*] Нажмите Ctrl+C для завершения и сохранения файла.\n")
    print("-" * 75)

    bytes_buffer = b''
    frame_count = 0  # Счетчик обработанных кадров

    try:
        while True:
            # Читаем поток по HTTP
            bytes_buffer += stream.read(4096)
            
            # Ищем границы JPEG-кадра
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg_data = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]

                # Декодируем байты в изображение
                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_count += 1  # Увеличиваем счетчик кадров

                # Детекция объектов через YOLO
                results = model.predict(
                    source=frame,
                    conf=0.3,
                    imgsz=WIDTH,
                    device=device,
                    verbose=False
                )

                # Вывод найденных объектов с указанием НОМЕРА КАДРА
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[cls_id]

                        print(f"[Кадр #{frame_count:<5}] Буква/Объект: {class_name:<10} | "
                              f"Точность: {confidence*100:.1f}% | "
                              f"Коорд: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

                # Наносим рамки детекции на кадр
                annotated_frame = results[0].plot()

                # Записываем кадр в файл MP4 на карту памяти
                out.write(annotated_frame)

    except KeyboardInterrupt:
        print("\n[*] Завершение работы пользователем...")

    finally:
        out.release()
        stream.close()
        print(f"[✔] Всего обработано кадров: {frame_count}")
        print(f"[✔] Файл '{OUTPUT_FILE}' успешно записан! Можно извлекать карту памяти.")

if __name__ == '__main__':
    main()