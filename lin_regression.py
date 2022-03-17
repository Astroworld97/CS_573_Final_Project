from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
import sys

brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")

mseList = []
rmseList = []
r2List = []
test = ""
try:
    for t in brazil_data["ticker"]:
        test = t
        data = brazil_data.loc[brazil_data["ticker"] == t]
        data = data.reset_index()
        features = ["close", "high", "low", "volumen"]
        new_data = pd.DataFrame(index=range(0, len(data)), columns=features)
        new_data["close"][0] = 0
        new_data["high"][0] = 0
        new_data["low"][0] = 0
        new_data["volumen"][0] = 0
        for i in range(1, len(data)):
            new_data["close"][i] = data["close"][i - 1]
            new_data["high"][i] = data["high"][i - 1]
            new_data["low"][i] = data["low"][i - 1]
            new_data["volumen"][i] = data["volumen"][i - 1]

        x = new_data[features]
        y = data["open"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        linreg = LinearRegression().fit(x_train, y_train)
        predictions = linreg.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, predictions)
        mseList.append(mse)
        rmse = math.sqrt(mse)
        rmseList.append(rmse)
        # print("MSE: " + str(mse))
        # print("RMSE: " + str(rmse))

        r2 = sklearn.metrics.r2_score(y_test, predictions)
        r2List.append(r2)
        # print("R^2: " + str(r2))
except:
    nothing = ""
mseNP = np.array(mseList)
mseSeries = pd.Series(mseNP)
rmseNP = np.array(rmseList)
rmseSeries = pd.Series(rmseNP)
r2NP = np.array(r2List)
r2Series = pd.Series(r2NP)
print(r2Series.mean())

# plt.figure(figsize=(10, 5))
# plt.plot(y_test)
# plt.plot(predictions)
# plt.show()

# gets number of rows and columns
# print(vale_data.shape)

# gets total number of cells
# print(vale_data.size)
