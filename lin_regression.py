from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
import sys

brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")

# start open
mseOpenList = []
rmseOpenList = []
r2OpenList = []
try:
    for t in brazil_data["ticker"]:
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
        mseOpenList.append(mse)
        rmse = math.sqrt(mse)
        rmseOpenList.append(rmse)
        # print("MSE: " + str(mse))
        # print("RMSE: " + str(rmse))

        r2 = sklearn.metrics.r2_score(y_test, predictions)
        r2OpenList.append(r2)
        # print("R^2: " + str(r2))
except:
    nothing = ""
mseOpenNP = np.array(mseOpenList)
mseOpenSeries = pd.Series(mseOpenNP)
rmseOpenNP = np.array(rmseOpenList)
rmseOpenSeries = pd.Series(rmseOpenNP)
r2OpenNP = np.array(r2OpenList)
r2OpenSeries = pd.Series(r2OpenNP)
print("Open average r^2: " + str(r2OpenSeries.mean()))
# end open

# start close
mseCloseList = []
rmseCloseList = []
r2CloseList = []
try:
    for t in brazil_data["ticker"]:
        data = brazil_data.loc[brazil_data["ticker"] == t]
        data = data.reset_index()
        features = ["open", "high", "low", "volumen"]
        new_data = pd.DataFrame(index=range(0, len(data)), columns=features)
        new_data["open"][0] = 0
        new_data["high"][0] = 0
        new_data["low"][0] = 0
        new_data["volumen"][0] = 0
        for i in range(1, len(data)):
            new_data["open"][i] = data["open"][i - 1]
            new_data["high"][i] = data["high"][i - 1]
            new_data["low"][i] = data["low"][i - 1]
            new_data["volumen"][i] = data["volumen"][i - 1]

        x = new_data[features]
        y = data["close"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        linreg = LinearRegression().fit(x_train, y_train)
        predictions = linreg.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, predictions)
        mseCloseList.append(mse)
        rmse = math.sqrt(mse)
        rmseCloseList.append(rmse)
        # print("MSE: " + str(mse))
        # print("RMSE: " + str(rmse))

        r2 = sklearn.metrics.r2_score(y_test, predictions)
        r2CloseList.append(r2)
        # print("R^2: " + str(r2))
except:
    nothing = ""
mseCloseNP = np.array(mseCloseList)
mseCloseSeries = pd.Series(mseCloseNP)
rmseCloseNP = np.array(rmseCloseList)
rmseCloseSeries = pd.Series(rmseCloseNP)
r2CloseNP = np.array(r2CloseList)
r2CloseSeries = pd.Series(r2CloseNP)
print("Close average r^2: " + str(r2CloseSeries.mean()))
# end close

# start high
mseHighList = []
rmseHighList = []
r2HighList = []
try:
    for t in brazil_data["ticker"]:
        data = brazil_data.loc[brazil_data["ticker"] == t]
        data = data.reset_index()
        features = ["open", "close", "low", "volumen"]
        new_data = pd.DataFrame(index=range(0, len(data)), columns=features)
        new_data["open"][0] = 0
        new_data["close"][0] = 0
        new_data["low"][0] = 0
        new_data["volumen"][0] = 0
        for i in range(1, len(data)):
            new_data["open"][i] = data["open"][i - 1]
            new_data["close"][i] = data["close"][i - 1]
            new_data["low"][i] = data["low"][i - 1]
            new_data["volumen"][i] = data["volumen"][i - 1]

        x = new_data[features]
        y = data["high"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        linreg = LinearRegression().fit(x_train, y_train)
        predictions = linreg.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, predictions)
        mseHighList.append(mse)
        rmse = math.sqrt(mse)
        rmseHighList.append(rmse)
        # print("MSE: " + str(mse))
        # print("RMSE: " + str(rmse))

        r2 = sklearn.metrics.r2_score(y_test, predictions)
        r2HighList.append(r2)
        # print("R^2: " + str(r2))
except:
    nothing = ""
mseHighNP = np.array(mseHighList)
mseHighSeries = pd.Series(mseHighNP)
rmseHighNP = np.array(rmseHighList)
rmseHighSeries = pd.Series(rmseHighNP)
r2HighNP = np.array(r2HighList)
r2HighSeries = pd.Series(r2HighNP)
print("High average r^2: " + str(r2HighSeries.mean()))
# end high

# start low
mseLowList = []
rmseLowList = []
r2LowList = []
try:
    for t in brazil_data["ticker"]:
        data = brazil_data.loc[brazil_data["ticker"] == t]
        data = data.reset_index()
        features = ["open", "close", "high", "volumen"]
        new_data = pd.DataFrame(index=range(0, len(data)), columns=features)
        new_data["open"][0] = 0
        new_data["close"][0] = 0
        new_data["high"][0] = 0
        new_data["volumen"][0] = 0
        for i in range(1, len(data)):
            new_data["open"][i] = data["open"][i - 1]
            new_data["close"][i] = data["close"][i - 1]
            new_data["high"][i] = data["high"][i - 1]
            new_data["volumen"][i] = data["volumen"][i - 1]

        x = new_data[features]
        y = data["low"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        linreg = LinearRegression().fit(x_train, y_train)
        predictions = linreg.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, predictions)
        mseLowList.append(mse)
        rmse = math.sqrt(mse)
        rmseLowList.append(rmse)
        # print("MSE: " + str(mse))
        # print("RMSE: " + str(rmse))

        r2 = sklearn.metrics.r2_score(y_test, predictions)
        r2LowList.append(r2)
        # print("R^2: " + str(r2))
except:
    nothing = ""
mseLowNP = np.array(mseLowList)
mseLowSeries = pd.Series(mseLowNP)
rmseLowNP = np.array(rmseLowList)
rmseLowSeries = pd.Series(rmseLowNP)
r2LowNP = np.array(r2LowList)
r2LowSeries = pd.Series(r2LowNP)
print("Low average r^2: " + str(r2LowSeries.mean()))
# end low

# start volumen
mseVList = []
rmseVList = []
r2VList = []
try:
    for t in brazil_data["ticker"]:
        data = brazil_data.loc[brazil_data["ticker"] == t]
        data = data.reset_index()
        features = ["open", "close", "high", "low"]
        new_data = pd.DataFrame(index=range(0, len(data)), columns=features)
        new_data["open"][0] = 0
        new_data["close"][0] = 0
        new_data["high"][0] = 0
        new_data["low"][0] = 0
        for i in range(1, len(data)):
            new_data["open"][i] = data["open"][i - 1]
            new_data["close"][i] = data["close"][i - 1]
            new_data["high"][i] = data["high"][i - 1]
            new_data["low"][i] = data["low"][i - 1]

        x = new_data[features]
        y = data["volumen"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        linreg = LinearRegression().fit(x_train, y_train)
        predictions = linreg.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, predictions)
        mseVList.append(mse)
        rmse = math.sqrt(mse)
        rmseVList.append(rmse)
        # print("MSE: " + str(mse))
        # print("RMSE: " + str(rmse))

        r2 = sklearn.metrics.r2_score(y_test, predictions)
        r2VList.append(r2)
        # print("R^2: " + str(r2))
except:
    nothing = ""
mseVNP = np.array(mseVList)
mseVSeries = pd.Series(mseVNP)
rmseVNP = np.array(rmseVList)
rmseVSeries = pd.Series(rmseVNP)
r2VNP = np.array(r2VList)
r2VSeries = pd.Series(r2VNP)
print("Vol. average r^2: " + str(r2VSeries.mean()))
# end volumen
