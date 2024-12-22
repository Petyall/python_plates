import cv2

from utils import load_model, load_image


def detect_plate(model, image, image_path, padding=10):
    """
    Распознавание границ автомобильного номера на фотографии
    и сохранение номера с максимальной вероятностью
    """
    results = model(image)

    best_plate = None
    max_confidence = -1

    for result in results:
        # Получение координат и вероятностей распознанных номеров
        boxes = result.boxes.xyxy.cpu().numpy()  # Координаты
        confidences = result.boxes.conf.cpu().numpy()  # Уверенности

        for box, confidence in zip(boxes, confidences):
            if confidence > max_confidence:
                max_confidence = confidence
                x1, y1, x2, y2 = map(int, box)

                # Добавление отступов с условием, чтобы они не выходили за границы изображения
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)

                # Сохранение текущего номера с наибольшей вероятностью
                best_plate = image[y1:y2, x1:x2]

    if best_plate is not None:
        # Сохранение изображения с наибольшей вероятностью
        cv2.imwrite(f'./data/detected_plates/{image_path[12:]}', best_plate)
        return True

    return False

if __name__ == "__main__":
    model = load_model("./trained_models/plate_detection.pt")

    # Обработка изображений
    for i in range(1, 16):
        image_path = f"./data/cars/{i}.jpg"
        image = load_image(image_path)

        if image is not None:
            success = detect_plate(model, image, image_path)
            if success:
                print(f"Изображение {i} обработано успешно.")
            else:
                print(f"На изображении {i} номера не обнаружены.")

    # # Обработка изображения
    # image_path = "./data/cars/2.jpg"
    # image = load_image(image_path)

    # if image is not None:
    #     success = detect_plate(model, image, image_path)
    #     if success:
    #         print(f"Изображение {image_path[12:]} обработано успешно.")
    #     else:
    #         print(f"На изображении {image_path[12:]} номера не обнаружены.")
