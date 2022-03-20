import numpy as np
import pandas as pd
from utilities import *
import tensorflow as tf
from evaluation import *

TICKER = [
    "VALE3",
    "PETR4",
    "MGLU3",
    "VVAR3",
    "ITUB4",
    "GOLL4",
    "LREN3",
    "WEGE3",
    "CTSA4",
    "EZTC3",
    "GGBR4",
    "MRFG3",
    "SULA11",
    "IRBR3",
    "TAEE11",
    "SUZB3",
]
LOOKBACK = 20
LOOKAHEAD = 2
LABEL_COL = "close"


def run_neural_net(TICKER, LOOKBACK, LOOKAHEAD, LABEL_COL="close"):

    # load data
    raw_df = get_ticker_data(TICKER)

    # get features
    raw_df = create_features(
        raw_df, lookback=LOOKBACK, label_col=LABEL_COL, lookahead=LOOKAHEAD
    )

    # extract features
    temp = extract_features_cols(raw_df, lookback=LOOKBACK)
    conv_df = None

    train_df, test_df = split_test_set(temp)
    normalize_cols(train_df, set(train_df.columns).difference({"time"}))
    normalize_cols(test_df, set(train_df.columns).difference({"time"}))
    y_train = train_df["label"]
    x_train = train_df.drop(["time", "label", "price", "class"], axis=1)
    x_test = test_df.drop(["time", "label", "price", "class"], axis=1)
    y_test = test_df["label"]
    print(train_df.columns)

    data_shape = np.shape(x_train)

    # build dense model
    dense_model = tf.keras.Sequential()
    dense_model.add(tf.keras.Input(shape=data_shape))
    dense_model.add(tf.keras.layers.Dense(units=40, activation="relu"))
    dense_model.add(tf.keras.layers.Dense(units=(20), activation="relu"))
    dense_model.add(tf.keras.layers.Dropout(0.3))
    dense_model.add(tf.keras.layers.Dense(units=(20), activation="relu"))
    # dense_model.add(tf.keras.layers.Dropout(0.3)) #uncomment for classification
    dense_model.add(tf.keras.layers.Dense(units=(10), activation="relu"))
    dense_model.add(tf.keras.layers.Dropout(0.3))
    dense_model.add(tf.keras.layers.Dense(units=(1), activation="relu"))
    # dense_model.add(tf.keras.layers.Dense(units=(1), activation='sigmoid')) #uncomment for classification

    print(dense_model.summary())

    # fit model
    dense_model.compile(
        optimizer="adam", loss="mean_squared_error", metrics="mean_absolute_error"
    )
    # dense_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy') #uncomment for classification
    dense_model.fit(
        x=x_train,
        y=y_train,
        validation_split=0.3,
        shuffle=True,
        batch_size=8,
        epochs=25,
    )

    # evaluate model
    print("Evaluating model")
    print(np.shape(x_test[str(LOOKBACK) + "_close"]))
    preds = dense_model.predict(x_test)
    # for i in range(len(preds)):
    #    print(preds[i], '|', y_test.iloc[i])
    plot_predictions(
        preds,
        y_test,
        save=False,
        name="regression" + str(LOOKBACK) + "_" + str(LOOKAHEAD) + ".png",
    )
    # for t in [0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]:
    #    print('Test acc:(', t,')', get_accuracy_diff(preds, y_test, t))
    # for t in [(0.3, 0.7),(0.4, 0.6),(0.2, 0.8),(0.45, 0.55)]:
    #    print('Test acc:(', t,')', get_accuracy_diff2(preds, y_test, t[0], t[1]))
    print(dense_model.evaluate(x_test, y_test, batch_size=1))
    # print(np.shape(preds), np.shape(y_test))
    return dense_model.evaluate(x_test, y_test, batch_size=1)
