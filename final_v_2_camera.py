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
    # print("распознавание номера началось")
    # frame = cv2.imread(image_path)
    frame = image_path
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
    # print("распознавание номера закончилось")


def rotate_plate(image_path):
    # image = cv2.imread(f'plates/{image_path}.jpg')
    # print("поворот начался")
    image = image_path
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
    # cv2.imwrite(f"plates/{image_path}_rotated.jpg", aligned_image)
    # print("поворот закончился")
    return aligned_image


def increase_resolution(img, scale_factor, image_path):
    print("скейл начался")
    height, width = img.shape[:2]
    if height < 100 or width < 200:
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # print("скейл закончился")
        return resized_img
    else:
        # print("скейл закончился")
        return img


def remove_background(image):
    # print("вырезание фона началось")
    img = image
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    # print("вырезание фона закончилось")

    return result

def adjust_brightness(image):
    # print("изменение яркости началось")
    mean_brightness = cv2.mean(image)[0]

    if mean_brightness > 127:
        adjusted_image = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    else:
        adjusted_image = cv2.convertScaleAbs(image, alpha=0.7, beta=30)

    # print("изменение яркости закончилось")
    return adjusted_image


def enhance_contrast(image):
    enhanced_image = cv2.equalizeHist(image)
    enhanced_image = cv2.medianBlur(enhanced_image, 3)
    enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    return enhanced_image


def binarize_image(image):
    _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)
    return processed_image


# def process_image_with_tesseract(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = increase_resolution(img, 2, image_path)
#     img = rotate_plate(img)

#     adjusted_image = adjust_brightness(img)
#     contrast_enhanced = enhance_contrast(adjusted_image)
#     background_removed = remove_background(contrast_enhanced)
#     binarized_image = binarize_image(background_removed)

#     text = pytesseract.image_to_string(binarized_image, config='--psm 10 -c tessedit_char_whitelist=ABEKkMmHOoPpCcTYyXx0123456789')
#     print(text)
#     return text


def process_image_with_tesseract(image_path):
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = increase_resolution(img, 2, image_path)
    # denoised_img = cv2.medianBlur(img, 5)
    # ret, binarized_img = cv2.threshold(denoised_img, 125, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(image_path, lang='eng', config='--psm 10 -c tessedit_char_whitelist=ABEKkMmHOoPpCcTYyXx0123456789')
    return text


def recognize_symbols_with_rotated_image(image_path):
    image = rotate_plate(image_path)
    text = process_image_with_tesseract(image)
    # print(text)
    return process_text(text, image)


def recognize_symbols_with_adjusted_image(image_path):
    image = adjust_brightness(image_path)
    text = process_image_with_tesseract(image)
    # print(text)
    return process_text(text, image)


def recognize_symbols_with_contrast_enhanced(image_path):
    image = enhance_contrast(image_path)
    text = process_image_with_tesseract(image)
    # print(text)
    return process_text(text, image)


def recognize_symbols_with_background_removed(image_path):
    image = remove_background(image_path)
    text = process_image_with_tesseract(image)
    # print(text)
    return process_text(text, image)


def recognize_symbols_with_binarized_image(image_path):
    image = binarize_image(image_path)
    text = process_image_with_tesseract(image)
    # print(text)
    return process_text(text, image)


def recognize_symbols_with_stock_image(image_path):
    text = process_image_with_tesseract(image_path)
    # print(text)
    return process_text(text, image_path)


def process_digits(text, digits, replacement):
    for digit in digits:
        if text[0] == digit:
            text = replacement + text[1:]
        if text[4] == digit:
            text = text[:4] + replacement + text[5:]
        if text[5] == digit:
            text = text[:5] + replacement + text[6:]
    return text


def process_text(text, image):
    if text:
        text = text.replace(" ", "").replace("\n", "")
        text = text.upper()
        if 8 <= len(text) <= 9:
            pattern = compile(r'^[A-Z]{1}\d{3}[A-Z]{2}\d{2,3}$')
            text = process_digits(text, '0', 'O')
            text = process_digits(text, ['2', '4'], 'A')
            text = process_digits(text, ['3', '8'], 'B')

            if pattern.match(text):
                return text, image
    return False, image


def recognize_symbols(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    symbols, image = recognize_symbols_with_stock_image(image)
    if not symbols:
        symbols_rotated, image = recognize_symbols_with_rotated_image(image)
        if not symbols_rotated:
            symbols_with_adjusted_image, image = recognize_symbols_with_adjusted_image(image)
            if not symbols_with_adjusted_image:
                symbols_with_contrast_enhanced, image = recognize_symbols_with_contrast_enhanced(image)
                if not symbols_with_contrast_enhanced:
                    symbols_with_background_removed, image = recognize_symbols_with_background_removed(image)
                    if not symbols_with_background_removed:
                        symbols_with_binarized_image, image = recognize_symbols_with_binarized_image(image)
                        if not symbols_with_binarized_image:
                            print('Символы на номере не распознаны')
                        else:
                            print(f'{symbols_with_binarized_image} - это российский автомобильный номер')
                    else:
                        print(f'{symbols_with_background_removed} - это российский автомобильный номер')
                else:
                    print(f'{symbols_with_contrast_enhanced} - это российский автомобильный номер')
            else:
                print(f'{symbols_with_adjusted_image} - это российский автомобильный номер')
        else:
            print(f'{symbols_rotated} - это российский автомобильный номер')
    else:
        print(f'{symbols} - это российский автомобильный номер')


def plate_recognition(image_path):
    find_plate = process_image(image_path)
    if find_plate:
        file = f'plates/{find_plate}.jpg'
        recognize_symbols(file)


# for i in range(1, 16):
#     time.sleep(1)
#     plate_recognition(f'cars/{i}.jpg')
    # find_plate = process_image(f'cars/{i}.jpg')
    # if find_plate:
    #     process_image_with_tesseract(f"plates/{find_plate}.jpg")



# cap = cv2.VideoCapture("rtsp://admin:vib32admin@192.168.1.108:554/RVi/1/1")
cap = cv2.VideoCapture(0)

time_start = datetime.now()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр.")
        break

    time_delta = datetime.now() - time_start
    if int(time_delta.seconds) >= 5:
        # print(time_delta)
        time_start = datetime.now()
        plate_recognition(frame)
        # file_name = process_image(frame)
        # if file_name:
        #     print("Обнаружен номер")
        #     img = increase_resolution(2, file_name)
        #     recognize_symbols(img)
            # rotate_plate(file_name)
            # delete_background(file_name)
            # recognize_symbols_with_rotated_image(file_name)
            # recognize_symbols_with_image_without_background(file_name)
            # recognize_symbols_with_stock_image(file_name)


cap.release()
cv2.destroyAllWindows()