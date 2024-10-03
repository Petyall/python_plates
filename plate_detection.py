import cv2

from ultralytics import YOLO


# Загрузка натренированной модели
model = YOLO('./trained_models/best.pt')

# Путь до фотографии
image_path = './cars/16.jpg'

# Поиск автомобильного номера на фотографии
results = model.predict(source=image_path, save=False)

# Загрузка оригинального изображения
img = cv2.imread(image_path)

# Параметр отступа пикселей на изображении (нужно, чтобы случайно не обрезать символы на номере)
padding = 10


# Цикл, перебирающий все распознанные номерные знаки на изображении
for result in results:
    # Получение координат ограничивающего прямоугольника (границы распознанного номера на изображении)
    boxes = result.boxes.xyxy.cpu().numpy()  # Перевод координат в массив

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Перевод координаты в int

        # Добавление отступов с условием, чтобы они не выходили за границы изображения
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        # Вырезание номерного знака с изображения
        plate = img[y1:y2, x1:x2]

        # Вывод автомобильного номера
        cv2.imshow('License Plate', plate)
        cv2.imwrite('16.jpg', plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
