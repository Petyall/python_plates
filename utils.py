import cv2

from ultralytics import YOLO


def load_model(model_path):
    """
    Загрузка модели для распознавания символов
    """
    return YOLO(model_path)


def load_image(image_path, *flags):
    """
    Загрузка изображения
    """
    image = cv2.imread(image_path, *flags)

    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")

    return image
