import os

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Путь к папке с данными для обучения
data_folder = "train_st"

# Список файлов в папке
files = os.listdir(data_folder)

for file in files:
    if file.endswith(".csv"):
        # Загружаем данные
        data = pd.read_csv(os.path.join(data_folder, file))

        # Обучение модели ARIMA
        model = ARIMA(data, order=(1, 1, 1))  # Параметры (p,d,q) выбираются опытным путем,
        # нужно более тщательно подобрать и проверить

        model_fit = model.fit()

        # Прогнозирование температуры на следующие 7 дней
        forecast = model_fit.forecast(steps=7)

        print(forecast)

        # Сохранение модели в файл
        # Сохранение модели
        model_filename = f"model_{file.split('.')[0]}.pkl"
        model_fit.save(model_filename)

        print(f"Модель для файла {file} сохранена как {model_filename}")
