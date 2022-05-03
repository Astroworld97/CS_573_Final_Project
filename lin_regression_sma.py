import numpy as np
import pandas as pd
from utilities import *
import tensorflow as tf
from evaluation import *
import pickle as pkl
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math


def run_lin_regression_sma(
    TICKER, LOOKBACK, LOOKAHEAD, moving_averages, LABEL_COL="close", gen_df=False
):
    if not gen_df:
        # save datasets
        with open("dataset/x_train.csv", "rb") as f:
            x_train = pkl.load(f)
        with open("dataset/x_test.csv", "rb") as f:
            x_test = pkl.load(f)
        with open("dataset/y_train.csv", "rb") as f:
            y_train = pkl.load(f)
        with open("dataset/y_test.csv", "rb") as f:
            y_test = pkl.load(f)
    else:
        # load data
        raw_df = get_ticker_data(TICKER)

        # get features
        raw_df = create_features(
            raw_df,
            moving_averages,
            lookback=LOOKBACK,
            label_col=LABEL_COL,
            lookahead=LOOKAHEAD,
        )

        # extract features
        temp = extract_features_cols(raw_df, moving_averages, lookback=LOOKBACK)

        train_df, test_df = split_test_set(temp, by_ticker=False, test_ticker="PETR4")
        normalize_cols(train_df, set(train_df.columns).difference({"time", "ticker"}))
        normalize_cols(test_df, set(train_df.columns).difference({"time", "ticker"}))
        y_train = np.asarray(train_df["label"]).astype("float32")
        x_train = train_df.drop(
            ["time", "label", "price", "class", "ticker", "diff"], axis=1
        )
        y_test = np.asarray(test_df["label"]).astype("float32")
        x_test = test_df.drop(
            ["time", "label", "price", "class", "ticker", "diff"], axis=1
        )

        suffix = ""
        for a in moving_averages:
            suffix += "_" + str(a)

        # save datasets
        with open("dataset/x_train_sma" + suffix + ".csv", "wb") as f:
            pkl.dump(x_train, f)
        with open("dataset/x_test_sma" + suffix + ".csv", "wb") as f:
            pkl.dump(x_test, f)
        with open("dataset/y_train_sma" + suffix + ".csv", "wb") as f:
            pkl.dump(y_train, f)
        with open("dataset/y_test_sma" + suffix + ".csv", "wb") as f:
            pkl.dump(y_test, f)

    data_shape = np.shape(x_train)

    linreg = LinearRegression().fit(x_train, y_train)

    # build dense model
    # dense_model = tf.keras.Sequential()
    # dense_model.add(tf.keras.Input(shape=data_shape))
    # dense_model.add(tf.keras.layers.Dense(units=40, activation="relu"))
    # dense_model.add(tf.keras.layers.Dense(units=(20), activation="relu"))
    # dense_model.add(tf.keras.layers.Dense(units=(20), activation="relu"))
    # dense_model.add(tf.keras.layers.Dense(units=(10), activation="relu"))
    # dense_model.add(tf.keras.layers.Dropout(0.3))
    # dense_model.add(tf.keras.layers.Dense(units=(1), activation="relu"))

    # print(dense_model.summary())

    # # fit model
    # dense_model.compile(
    #     optimizer="adam", loss="mean_squared_error", metrics="mean_absolute_error"
    # )
    # # dense_model.compile(optimizer='adam', loss=loss, metrics='mean_absolute_error')
    # dense_model.fit(
    #     x=x_train,
    #     y=y_train,
    #     validation_split=0.3,
    #     shuffle=True,
    #     batch_size=16,
    #     epochs=30,
    # )

    # evaluate model
    print("Evaluating model", moving_averages)
    predictions = linreg.predict(x_test)
    # preds = dense_model.predict(x_test)
    # plot_predictions(
    #     predictions,
    #     y_test,
    #     name="regression" + str(LOOKBACK) + "_" + str(LOOKAHEAD) + ".png",
    #     save=False,
    # )
    # plot_predictions(
    #     preds[200:400],
    #     y_test[200:400],
    #     name="regression" + str(LOOKBACK) + "_" + str(LOOKAHEAD) + ".png",
    #     save=False,
    # )
    mae = sklearn.metrics.mean_absolute_error(y_test, predictions)
    print("MAE: " + str(mae))
    mse = sklearn.metrics.mean_squared_error(y_test, predictions)
    print("MSE: " + str(mse))
    rmse = math.sqrt(mse)
    print("RMSE: " + str(rmse))
    r2 = sklearn.metrics.r2_score(y_test, predictions)
    print("r^2: " + str(r2))
    plot_predictions(predictions, y_test, True, "test_plot.png")
    # print(dense_model.evaluate(x_test, y_test, batch_size=1))
    print("Accuracy:", get_accuracy(predictions, y_test))
    # return dense_model.evaluate(x_test, y_test, batch_size=1)
    return mae


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
