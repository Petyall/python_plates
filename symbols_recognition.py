import cv2

from ultralytics import YOLO


# Загрузка натренированной модели
model = YOLO("./trained_models/symbols_detection.pt")

# Путь до фотографии и ее чтение
image_path = "./detected_plates/5_license_plate.jpg"
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

# # Обрезка и сохранение символов
# for box, confidence, class_id in sorted_boxes:
#     x_center, y_center, width, height = box
#     class_name = model.names[int(class_id)]
#     print(f"Detected {class_name} with confidence {confidence:.2f} at [{x_center}, {y_center}, {width}, {height}]")
    
#     x1 = int(x_center - width / 2)
#     y1 = int(y_center - height / 2)
#     x2 = int(x_center + width / 2)
#     y2 = int(y_center + height / 2)
    
#     symbol_image = image[y1:y2, x1:x2]
#     cv2.imwrite(f"symbol_{int(class_id)}.jpg", symbol_image)
