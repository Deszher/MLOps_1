import os

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Путь к папке с тестовыми данными
test_data_folder = 'test_st/'

# Считывание данных из всех файлов в папке test
for file in os.listdir(test_data_folder):
    if file.endswith('.csv'):
        data = pd.read_csv(os.path.join(test_data_folder, file))

        # Преобразование данных во врем. ряд
        time_series = pd.Series(data.values.flatten())

        # Загрузка сохраненной модели ARIMA
        model_filename = f'model_data_train_{file.split("_")[-1]}'
        model = joblib.load(f'{model_filename[:-4]}.pkl')

        # Прогнозирование с помощью ARIMA
        arima_model = ARIMA(data, order=(model.model_orders['ar'], 1, model.model_orders['ma']))
        fitted_model = arima_model.fit()
        forecast = fitted_model.forecast(steps=len(data))

        # Рассчитаем метрики(MSE и MAE)
        mse = mean_squared_error(data, forecast)
        mae = mean_absolute_error(data, forecast)

        # Вывод результата прогнозирования и метрик
        print(f'Прогноз для файла {file}: {forecast}')
        print(f'MSE: {mse}')
        print(f'MAE: {mae}')
