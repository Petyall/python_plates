import cv2
import numpy as np

from scipy import ndimage
from PIL import Image as im
from utils import load_image


def resize_to_dpi(image, target_dpi=300, current_dpi=96):
    """
    Изменение размера изображения для достижения необходимого DPI
    """
    scale_factor = target_dpi / current_dpi
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def binarize_image(image, threshold=127):
    """
    Преобразование изображения в бинарное
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image


def calculate_score(arr, angle):
    """
    Вычисление оценки для заданного угла на основе гистограммы проекций вдоль оси
    """
    rotated = ndimage.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(rotated, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)

    return score


def find_optimal_angle(binary_image, delta=1, limit=5):
    """
    Нахождение оптимального угла для коррекции наклона изображения
    """
    angles = np.arange(-limit, limit + delta, delta)
    scores = []

    for angle in angles:
        score = calculate_score(binary_image, angle)
        scores.append(score)

    best_angle = angles[np.argmax(scores)]

    return best_angle


def correct_skew(image, angle):
    """
    Корректирование наклона изображения на указанный угол
    """
    return ndimage.rotate(image, angle, reshape=False, order=0)


def apply_morphological_operations(image):
    """
    Применение эрозии, дилатации и сглаживания к изображению
    """
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    smoothed = cv2.GaussianBlur(dilated, (5, 5), 0)

    return smoothed


def save_image(image, filename):
    """
    Сохранение изображения в файл
    """
    im.fromarray(image).save(filename)

    return True


def image_preprocess(image_path: str):
    """
    Предобработка изображения перед распознаванием символов
    """
    # 1. Чтение изображения
    image = load_image(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Изменение размера изображения
    # https://tesseract-ocr.github.io/tessdoc/ImproveQuality#:~:text=a%20DPI%20of-,at%20least%20300%20dpi,-%2C%20so%20it%20may
    resized_image = resize_to_dpi(image, target_dpi=300)

    # 3. Бинаризация изображения
    binarized_image = binarize_image(resized_image)

    # 4. Коррекция угла наклона текста
    # https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7#:~:text=2.-,Skew%20Correction,-%3A%20While%20scanning
    best_angle = find_optimal_angle(binarized_image)
    corrected_image = correct_skew(binarized_image, best_angle)

    # 5. Морфологическая обработка изображения
    final_image = apply_morphological_operations(corrected_image)

    # 6. Сохранение результата
    save_image(final_image, f'./data/preprocessed_plates/{image_path[23:]}')


if __name__ == "__main__":
    # Предобработка изображений
    for i in range (1, 16):
        image_preprocess(f"./data/detected_plates/{i}.jpg")

    # Предобработка изображения
    # image_preprocess("./data/detected_plates/1.jpg")
