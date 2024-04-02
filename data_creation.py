import os

import numpy as np

# Создаем папки "train" и "test", если они не существуют
if not os.path.exists("train"):
    os.makedirs("train")
if not os.path.exists("test"):
    os.makedirs("test")


# Генерируем данные дневной температуры
def generate_temperature_data(days):
    base_temperature = 25
    temperature_variation = np.random.normal(0, 5, days)  # +- в пределах 5 градусов к базовой температуре

    # Создаем температурные аномалии
    anomaly_indices = np.random.choice(days, int(days * 0.1), replace=False)  # случайные индексы
    temperature_variation[anomaly_indices] += np.random.uniform(-5, 10, len(anomaly_indices))

    # Добавляем шум
    noise = np.random.normal(0, 1, days)

    temperature_data = base_temperature + temperature_variation + noise

    return temperature_data.astype(int)


# Создаем нужное количество наборов данных, содержащих заданное количество строк
days_train = 90
days_test = 10
datasets = 2

for i in range(datasets):
    temperature_data = generate_temperature_data(days_train)
    np.savetxt(f"train/data_train_{i}.csv", temperature_data, delimiter=',', fmt='%.1f')

for i in range(datasets):
    temperature_data = generate_temperature_data(days_test)
    np.savetxt(f"test/data_test_{i}.csv", temperature_data, delimiter=',', fmt='%.1f')
