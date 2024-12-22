from utils import load_model, load_image


def recognite_symbols(model, image):
    """
    Применение модели к изображению и возврат распознанных символов
    """
    results = model(image)

    # Получение распознанных объектов (символов)
    boxes = results[0].boxes
    confidences = boxes.conf
    classes = boxes.cls
    
    # Сортировка детекции по координате x (распознавание символов слева направо)
    sorted_boxes = sorted(zip(boxes.xywh, confidences, classes), key=lambda x: x[0][0])
    
    return sorted_boxes


def generate_plate_number(sorted_boxes, model):
    """
    Генерация строки номера из результатов детектирования символов
    """
    symbols = []
    
    for box_info in sorted_boxes:
        # xywh - информация о позиции символа, confidence - уверенности, class_id - классе символа
        xywh, confidence, class_id = box_info
        symbol_name = model.names[int(class_id)]
        symbols.append(symbol_name)
    
    return ''.join(symbols)


if __name__ == "__main__":
    model = load_model("./trained_models/symbols_detection.pt")
    
    # Обработка изображений
    for i in range(1, 16):
        image_path = f"./data/preprocessed_plates/{i}.jpg"
        image = load_image(image_path)
        
        if image is not None:
            sorted_boxes = recognite_symbols(model, image)
            plate_number = generate_plate_number(sorted_boxes, model)
            
            print(f"Распознанный номер: {plate_number}")

    # # Обработка изображения
    # image_path = "./data/preprocessed_plates/1.jpg"
    # image = load_image(image_path)
    
    # if image is not None:
    #     sorted_boxes = detect_symbols(model, image)
    #     plate_number = generate_plate_number(sorted_boxes, model)
        
    #     print(f"Распознанный номер: {plate_number}")
