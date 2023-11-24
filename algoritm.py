import pytesseract
import cv2
import imutils
import numpy as np
# Устанавливаем путь к исполняемому файлу Tesseract OCR для распознавания текста на изображениях
pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'

# Загружаем предварительно обученный классификатор Haar Cascade для распознавания российских автомобильных номерных знаков
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Загружаем изображение с диска и преобразуем его в оттенки серого
img = cv2.imread('plates/plate.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применяем двусторонний фильтр для сглаживания изображения без потери краев, затем используем детектор краев Canny
img_filter = cv2.bilateralFilter(gray, 11, 17, 17)

# Масштабируем изображение
img_filter = cv2.resize(img_filter, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

# Затем используем детектор краев Canny
edges = cv2.Canny(img_filter, 30, 200)

# Применяем адаптивную пороговую обработку
thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# Находим контуры на обработанном изображении
cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)

# Пытаемся найти контур, который имеет 4 угла (предположительно номерной знак)
pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True)
    if len(approx) == 4:
        pos = approx
        break

# Создаем маску для выделения области с номерным знаком
mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

# Извлекаем координаты области с номерным знаком
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
crop = gray[x1:x2 + 1, y1:y2 + 1]

# Применяем каскадный классификатор к изображению для обнаружения номерных знаков
plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Проверяем, найдены ли номерные знаки на изображении
if len(plates) > 0:
    # Выбираем первый найденный номерной знак и вырезаем его из изображения
    (x, y, w, h) = plates[0]
    crop = gray[y:y+h, x:x+w]
    
    # Применяем пороговую обработку для улучшения распознавания текста
    thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('image', thresh)
    # cv2.waitKey()
    
    # Используем Tesseract OCR для распознавания текста на номерном знаке
    recognized_text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 3 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789')
    
    # Выводим распознанный текст номерного знака
    print("Распознанный текст номерного знака:", recognized_text)
    
else:
    # Выводим сообщение, если номерные знаки не были найдены
    print("Номерные знаки не найдены.")
