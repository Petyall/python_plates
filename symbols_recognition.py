import cv2

from ultralytics import YOLO


# Загрузка натренированной модели
model = YOLO("./trained_models/symbols_detection.pt")

for i in range(1, 16):
    # Путь до фотографии и ее чтение
    image_path = f"./data/preprocessed_plates/{i}.jpg"
    image = cv2.imread(image_path)

    # Применение модели на изображении
    results = model(image)

    # Получение распознанных объектов (символов)
    boxes = results[0].boxes
    confidences = boxes.conf
    classes = boxes.cls

    # Сортировка детекции по координате x (распознавание символов слева направо)
    sorted_boxes = sorted(zip(boxes.xywh, confidences, classes), key=lambda x: x[0][0])

    # Генерация строки номера
    plate_number = ''.join([model.names[int(cls)] for _, _, cls in sorted_boxes])

    print(f"Распознанный номер: {plate_number}")
