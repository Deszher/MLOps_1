import os

import numpy as np
from sklearn.preprocessing import StandardScaler

if not os.path.exists("train_st"):
    os.makedirs("train_st")
if not os.path.exists("test_st"):
    os.makedirs("test_st")

i = 0
while os.path.exists(f"train/data_train_{i}.csv") and os.path.exists(f"test/data_test_{i}.csv"):
    temperature_data_train = np.loadtxt(f"train/data_train_{i}.csv")
    temperature_data_test = np.loadtxt(f"test/data_test_{i}.csv")

    # Применение StandardScaler
    scaler = StandardScaler()
    temperature_data_train_scaled = scaler.fit_transform(temperature_data_train.reshape(-1, 1)).flatten()
    temperature_data_test_scaled = scaler.transform(temperature_data_test.reshape(-1, 1)).flatten()

    np.savetxt(f"train_st/data_train_st{i}.csv", temperature_data_train_scaled, delimiter=',')
    np.savetxt(f"test_st/data_test_st{i}.csv", temperature_data_test_scaled, delimiter=',')

    i += 1
