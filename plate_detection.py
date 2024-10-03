from ultralytics import YOLO


# Загрузка натренированной модели
model = YOLO('./trained_models/best.pt')

# Поиск контура автомобильного номера на фотографии
results = model.predict(source='./cars/1.jpg')

# Вывод результатов
for result in results:
    result.show()
