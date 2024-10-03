import cv2
import numpy as np


def background_removing(image):
    # Применение пороговой обработки
    _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Поиск контуров
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Предположение, что номерной знак - самый крупный прямоугольный контур на изображении
    license_plate_contour = max(contours, key=cv2.contourArea)

    # Создание маски
    mask = np.zeros_like(image)

    # Заполнение маски белым внутри контуров номерного знака
    cv2.drawContours(mask, [license_plate_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Наложение маски на изображение
    result = cv2.bitwise_and(image, mask)

    # Показать исходное изображение и результат
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Image with Background Removed', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result


def adjust_threshold_by_brightness(image):    
    # Анализ яркости: вычислить среднюю яркость
    mean_brightness = np.mean(image)
    
    # Настройка параметров в зависимости от яркости
    if mean_brightness < 50:
        # Темное изображение, уменьшить порог и фильтрацию
        threshold_value = 80
        blur_kernel = (3, 3)
    elif mean_brightness < 150:
        # Средняя яркость, использовать стандартные параметры
        threshold_value = 120
        blur_kernel = (5, 5)
    else:
        # Светлое изображение, увеличить порог и размытие
        threshold_value = 160
        blur_kernel = (7, 7)
    
    # Применение размытия для уменьшения шума
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    
    # Применение пороговой обработки
    _, threshold = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # print(f"Средняя яркость изображения: {brightness}")
    # cv2.imshow('Threshold Image', threshold_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return threshold


def preprocess_image(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)

    # Перевод в черно-белое изображение
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Удаление шумов при помощи размытия
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    background_remove = background_removing(blurred)

    binary = adjust_threshold_by_brightness(background_remove)

    # Шаг 6: Эрозия и дилатация
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Шаг 7: Сглаживание
    smoothed = cv2.GaussianBlur(dilated, (5, 5), 0)

    return smoothed


processed_image = preprocess_image('./16.jpg')

cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
