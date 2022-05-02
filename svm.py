from sklearn.svm import SVR
from sklearn.svm import LinearSVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
from utilities import *
from evaluation import *


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

    # data_shape = np.shape(x_train)
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
