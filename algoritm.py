import os
import cv2
import time
import pytesseract
from datetime import datetime

# pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'

def capture_image():
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # filename = "captured_images/{}.png".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # cv2.imwrite(filename, frame)
    # cap.release()
    # return filename
    filename = 'captured_images/plate.jpg'
    return filename

def carplate_extract(image, carplate_haar_cascade):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    if len(carplate_rects) == 0:
        return None

    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10, x+15:x+w-20]

    return carplate_img


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized_image


def main():
    while True:

        filename = capture_image()
        carplate_img_rgb = cv2.imread(filename)
        carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

        carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)

        if carplate_extract_img is not None:
            carplate_extract_img = enlarge_img(carplate_extract_img, 150)
            carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)

            print('Номер авто: ', pytesseract.image_to_string(
                carplate_extract_img_gray,
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789')
            )

        time.sleep(1)


if __name__ == '__main__':
    main()