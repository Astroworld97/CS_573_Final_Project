from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
from utilities import *
from evaluation import *


def run_lin_regression(TICKER, LOOKBACK, LOOKAHEAD, LABEL_COL="close"):

    raw_df = get_ticker_data(TICKER)
    raw_df = create_features(raw_df, lookback=LOOKBACK, label_col=LABEL_COL)
    temp = extract_features_cols(raw_df, lookback=LOOKBACK)
    train_df, test_df = split_test_set(
        temp
    )  # the test set is only from the last year. The training set is all other years.
    normalize_cols(train_df, set(train_df.columns).difference({"time"}))
    normalize_cols(test_df, set(train_df.columns).difference({"time"}))
    y_train = train_df["label"]
    x_train = train_df.drop(["time", "label", "price", "class"], axis=1)
    x_test = test_df.drop(["time", "label", "price", "class"], axis=1)
    y_test = test_df["label"]
    # print(train_df.columns)

    data_shape = np.shape(x_train)

    # start close
    # data = brazil_data.loc[brazil_data["VALE3"]]
    # features = ["open", "high", "low", "volumen"]
    # new_data = pd.DataFrame(index=range(0, len(data)), columns=features)
    # new_data["open"][0] = 0
    # new_data["high"][0] = 0
    # new_data["low"][0] = 0
    # new_data["volumen"][0] = 0
    # for i in range(1, len(data)):
    #     new_data["open"][i] = data["open"][i - 1]
    #     new_data["high"][i] = data["high"][i - 1]
    #     new_data["low"][i] = data["low"][i - 1]
    #     new_data["volumen"][i] = data["volumen"][i - 1]

    linreg = LinearRegression().fit(x_train, y_train)
    predictions = linreg.predict(x_test)

    mae = sklearn.metrics.mean_absolute_error(y_test, predictions)
    print("MAE: " + str(mae))
    mse = sklearn.metrics.mean_squared_error(y_test, predictions)
    print("MSE: " + str(mse))
    rmse = math.sqrt(mse)
    print("RMSE: " + str(rmse))
    r2 = sklearn.metrics.r2_score(y_test, predictions)
    print("r^2: " + str(r2))
    plot_predictions(predictions, y_test, True, "test_plot.png")
    # end close
