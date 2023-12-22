import cv2
import numpy as np
import time
from datetime import datetime
from re import compile
from imutils import grab_contours
from pytesseract import pytesseract

cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'

def process_image(image_path):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_filter = cv2.bilateralFilter(gray, 11, 17, 17)
    img_filter = cv2.resize(img_filter, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    edges = cv2.Canny(gray, 30, 200)
    cont = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = grab_contours(cont)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)
    pos = None
    for c in cont:
        approx = cv2.approxPolyDP(c, 10, True)
        if len(approx) == 4:
            pos = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    if pos is not None:
        new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
        bitwise_img = cv2.bitwise_and(frame, frame, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1, x2, y2) = (0, 0, 0, 0) 
        if len(x) > 0 and len(y) > 0:
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            crop = gray[x1:x2 + 1, y1:y2 + 1]
            plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(plates) > 0:
                (x, y, w, h) = plates[0]
                crop = gray[y:y+h, x:x+w]
                time_now = datetime.now().strftime("%H_%M_%S")
                file_name = f"plate_{time_now}"
                cv2.imwrite(f"plates/{file_name}.jpg", crop)
                return file_name
            else:
                print("Номерной знак не обнаружен.")
        else:
            print("Контуры не найдены. Невозможно вырезать область с номерным знаком.")
    else:
        print("Контуры не найдены. Невозможно создать маску для номерного знака.")


def rotate_plate(image_path):
    image = cv2.imread(f'plates/{image_path}.jpg')
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    x1, y1, x2, y2 = 0, 0, 0, 0

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if theta > np.pi / 180 * 45 and theta < np.pi / 180 * 135:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    cv2.imwrite(f"plates/{image_path}_rotated.jpg", aligned_image)


def delete_background(image_path):
    image = cv2.imread(f'plates/{image_path}_rotated.jpg')
    # result = rembg.remove(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"plates/{image_path}_without_background.jpg", result)


def process_image_with_tesseract(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    denoised_img = cv2.medianBlur(img, 5)
    ret, binarized_img = cv2.threshold(denoised_img, 125, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Aligned Image', binarized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    text = pytesseract.image_to_string(binarized_img, lang='eng', config='--psm 10 -c tessedit_char_whitelist=ABEKkMmHOoPpCcTYyXx0123456789')
    # text = pytesseract.image_to_string(binarized_img, lang='rus', config='--psm 10 -c tessedit_char_whitelist=АаВвЕеКкМмНнОоРрСсТтУуХх0123456789')
    return text


def recognize_symbols_with_rotated_image(image_path):
    rotate_plate(image_path)
    text = process_image_with_tesseract(f'plates/{image_path}_rotated.jpg')
    # print(text)
    return process_text(text)


def recognize_symbols_with_image_without_background(image_path):
    delete_background(image_path)
    text = process_image_with_tesseract(f'plates/{image_path}_without_background.jpg')
    # print(text)
    return process_text(text)


def recognize_symbols_with_stock_image(image_path):
    text = process_image_with_tesseract(f'plates/{image_path}.jpg')
    # print(text)
    return process_text(text)


def process_digits(text, digits, replacement):
    for digit in digits:
        if text[0] == digit:
            text = replacement + text[1:]
        if text[4] == digit:
            text = text[:4] + replacement + text[5:]
        if text[5] == digit:
            text = text[:5] + replacement + text[6:]
    return text


def process_text(text):
    if text:
        text = text.replace(" ", "").replace("\n", "")
        text = text.upper()
        if 8 <= len(text) <= 9:
            pattern = compile(r'^[A-Z]{1}\d{3}[A-Z]{2}\d{2,3}$')
            text = process_digits(text, '0', 'O')
            text = process_digits(text, ['2', '4'], 'A')
            text = process_digits(text, ['3', '8'], 'B')

            if pattern.match(text):
                return text
    return False


def recognize_symbols(image_path):
    symbols = recognize_symbols_with_stock_image(image_path)
    if not symbols:
        symbols_rotated = recognize_symbols_with_rotated_image(image_path)
        if not symbols_rotated:
            symbols_without_background = recognize_symbols_with_image_without_background(image_path)
            if not symbols_without_background:
                print('Символы на номере не распознаны')
            else:
                print(f'{symbols_without_background} - это российский автомобильный номер')
        else:
            print(f'{symbols_rotated} - это российский автомобильный номер')
    else:
        print(f'{symbols} - это российский автомобильный номер')


def plate_recognition(image_path):
    find_plate = process_image(image_path)
    if find_plate:
        recognize_symbols(find_plate)


# plate = f'cars/17.jpg'
# plate_recognition(plate)

for i in range(1, 16):
    time.sleep(1)
    plate_recognition(f'cars/{i}.jpg')