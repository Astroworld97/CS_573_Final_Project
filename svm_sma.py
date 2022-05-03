import numpy as np
import pandas as pd
from utilities import *
import tensorflow as tf
from evaluation import *
import pickle as pkl
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math


def run_svm_sma(
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

    # build dense model
    data_shape = np.shape(x_train)
    c = 100
    svr_rbf = SVR(kernel="rbf", C=c)
    svr_rbf.fit(x_train, y_train)
    svr_poly = SVR(kernel="poly", C=c)
    svr_poly.fit(x_train, y_train)

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
    predictions = svr_rbf.predict(x_test)
    # preds = dense_model.predict(x_test)
    # plot_predictions(
    #     preds,
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
    # print(dense_model.evaluate(x_test, y_test, batch_size=1))
    print("Accuracy:", get_accuracy(predictions, y_test))
    mae_rbf = sklearn.metrics.mean_absolute_error(y_test, predictions)
    # svr_rbf_confidence = svr_rbf.score(y_test, predictions)
    print("svr_rbf MAE: " + str(mae_rbf) + "for a C parameter of: " + str(100))
    return mae_rbf
    # Uncomment to plot:# plot_predictions(predictions, y_test, True, "svr_rbf" + str(c))
    # return dense_model.evaluate(x_test, y_test, batch_size=1)
    # predictions = svr_poly.predict(x_test)
    # mae_poly = sklearn.metrics.mean_absolute_error(y_test, predictions)
    #     print(
    #     "svr_poly confidence MAE: "
    #     + str(mae_poly)
    #     + "for a C parameter of: "
    #     + str(100)
    # )
    # Uncomment to plot: #plot_predictions(predictions, y_test, True, "svr_poly" + str(c))


def run_svm(TICKER, LOOKBACK, LOOKAHEAD, LABEL_COL="close"):

    # data management start
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
    # data management done

    data_shape = np.shape(x_train)
    c = 100
    svr_rbf = SVR(kernel="rbf", C=c)
    svr_rbf.fit(x_train, y_train)
    # x_test = x_test.to_numpy()
    # x_test = x_test.reshape(-1, 26)
    predictions = svr_rbf.predict(x_test)
    # y_test = y_test.to_numpy()
    # y_test = y_test.reshape(-1, 26)
    mae_rbf = sklearn.metrics.mean_absolute_error(y_test, predictions)
    # svr_rbf_confidence = svr_rbf.score(y_test, predictions)
    print(
        "svr_rbf confidence, aka R^2: "
        + str(mae_rbf)
        + "for a C parameter of: "
        + str(100)
    )
    plot_predictions(predictions, y_test, True, "svr_rbf" + str(c))

    svr_poly = SVR(kernel="poly", C=c)
    svr_poly.fit(x_train, y_train)
    # x_test = x_test.to_numpy()
    # x_test = x_test.reshape(-1, 26)
    predictions = svr_poly.predict(x_test)
    # y_test = y_test.to_numpy()
    # y_test = y_test.reshape(-1, 26)
    # svr_poly_confidence = svr_poly.score(y_test, predictions)
    mae_poly = sklearn.metrics.mean_absolute_error(y_test, predictions)
    print(
        "svr_poly confidence, aka R^2: "
        + str(mae_poly)
        + "for a C parameter of: "
        + str(100)
    )
    plot_predictions(predictions, y_test, True, "svr_poly" + str(c))
