import numpy as np
import pandas as pd
from utilities import *
import tensorflow as tf
from evaluation import *
import pickle as pkl

#TODO: get diff based on lookahead value

TICKER=['VALE3', 'PETR4', 'MGLU3', 'VVAR3', 'ITUB4', 'GOLL4', 'LREN3', 'WEGE3', 'CTSA4', 'EZTC3', 'GGBR4',
            'MRFG3', 'SULA11', 'IRBR3', 'TAEE11', 'SUZB3']
LOOKBACK=20
LOOKAHEAD=2
LABEL_COL='close'

def run_neural_net_regression(TICKER, LOOKBACK, LOOKAHEAD, moving_averages, LABEL_COL='close', gen_df=False):
    if(not gen_df):
        #save datasets
        with open('dataset/x_train.csv', 'rb') as f:
            x_train=pkl.load(f)
        with open('dataset/x_test.csv', 'rb') as f:
            x_test=pkl.load(f)
        with open('dataset/y_train.csv', 'rb') as f:
            y_train=pkl.load(f)
        with open('dataset/y_test.csv', 'rb') as f:
            y_test=pkl.load(f)
    else:
        #load data
        raw_df=get_ticker_data(TICKER)

        #get features
        raw_df=create_features(raw_df, moving_averages, lookback=LOOKBACK, label_col=LABEL_COL, lookahead=LOOKAHEAD)

        #extract features
        temp=extract_features_cols(raw_df, moving_averages, lookback=LOOKBACK)

        train_df, test_df = split_test_set(temp, by_ticker=False, test_ticker='PETR4')
        normalize_cols(train_df, set(train_df.columns).difference({'time', 'ticker'}))
        normalize_cols(test_df, set(train_df.columns).difference({'time', 'ticker'}))
        y_train=np.asarray(train_df['label']).astype('float32')
        x_train=train_df.drop(['time', 'label', 'price', 'class', 'ticker', 'diff'], axis=1)
        y_test=np.asarray(test_df['label']).astype('float32')
        x_test=test_df.drop(['time', 'label', 'price', 'class', 'ticker', 'diff'], axis=1)

        suffix=''
        for a in moving_averages:
            suffix+='_'+str(a)

        #save datasets
        with open('dataset/x_train_sma'+suffix+'.csv', 'wb') as f:
            pkl.dump(x_train, f)
        with open('dataset/x_test_sma'+suffix+'.csv', 'wb') as f:
            pkl.dump(x_test, f)
        with open('dataset/y_train_sma'+suffix+'.csv', 'wb') as f:
            pkl.dump(y_train, f)
        with open('dataset/y_test_sma'+suffix+'.csv', 'wb') as f:
            pkl.dump(y_test, f)
    

    data_shape=np.shape(x_train)

    #build dense model
    dense_model=tf.keras.Sequential()
    dense_model.add(tf.keras.Input(shape=data_shape))
    dense_model.add(tf.keras.layers.Dense(units=40, activation='relu'))
    dense_model.add(tf.keras.layers.Dense(units=(20), activation='relu'))
    dense_model.add(tf.keras.layers.Dense(units=(20), activation='relu'))
    dense_model.add(tf.keras.layers.Dense(units=(10), activation='relu'))
    dense_model.add(tf.keras.layers.Dropout(0.3))
    dense_model.add(tf.keras.layers.Dense(units=(1), activation='relu'))

    print(dense_model.summary())


    #fit model
    dense_model.compile(optimizer='adam', loss='mean_squared_error', metrics='mean_absolute_error')
    #dense_model.compile(optimizer='adam', loss=loss, metrics='mean_absolute_error')
    dense_model.fit(x=x_train, y=y_train, validation_split=0.3, shuffle=True, batch_size=16, epochs=30)

    #evaluate model
    print('Evaluating model', moving_averages)
    preds=dense_model.predict(x_test)
    plot_predictions(preds, y_test, name='regression'+str(LOOKBACK)+'_'+str(LOOKAHEAD)+'.png', save=False)
    plot_predictions(preds[200:400], y_test[200:400], name='regression'+str(LOOKBACK)+'_'+str(LOOKAHEAD)+'.png', save=False)
    print(dense_model.evaluate(x_test, y_test, batch_size=1))
    print('Accuracy:', get_accuracy(preds, y_test))
    return dense_model.evaluate(x_test, y_test, batch_size=1)

def run_neural_net_regression_diff(TICKER, LOOKBACK, LOOKAHEAD, LABEL_COL='close'):
    #load datasets
    #with open('x_train_diff.csv', 'rb') as f:
    #    x_train=pkl.load(f)
    #with open('x_test_diff.csv', 'rb') as f:
    #    x_test=pkl.load(f)
    #with open('y_train_diff.csv', 'rb') as f:
    #    y_train=pkl.load(f)
    #with open('y_test_diff.csv', 'rb') as f:
    #    y_test=pkl.load(f)

    #load data
    raw_df=get_ticker_data(TICKER)

    #get features
    raw_df=create_features(raw_df, lookback=LOOKBACK, label_col=LABEL_COL, lookahead=LOOKAHEAD)

    #extract features
    temp=extract_features_cols(raw_df, lookback=LOOKBACK)

    train_df, test_df = split_test_set(temp, by_ticker=True, test_ticker='PETR4')
    normalize_cols(train_df, set(train_df.columns).difference({'time', 'ticker', 'diff'}))
    normalize_cols(test_df, set(train_df.columns).difference({'time', 'ticker', 'diff'}))
    standardize_cols(train_df, {'diff'})
    standardize_cols(test_df, {'diff'})
    y_train=np.asarray(train_df['diff']).astype('float32')
    x_train=train_df.drop(['time', 'label', 'price', 'class', 'ticker', 'diff'], axis=1)
    y_test=np.asarray(test_df['diff']).astype('float32')
    x_test=test_df.drop(['time', 'label', 'price', 'class', 'ticker', 'diff'], axis=1)

    #save datasets
    with open('x_train_diff_5.csv', 'wb') as f:
        pkl.dump(x_train, f)
    with open('x_test_diff_5.csv', 'wb') as f:
        pkl.dump(x_test, f)
    with open('y_train_diff_5.csv', 'wb') as f:
        pkl.dump(y_train, f)
    with open('y_test_diff_5.csv', 'wb') as f:
        pkl.dump(y_test, f)
    
    data_shape=np.shape(x_train)
    print(data_shape)
    temp_list=[1]
    for i in data_shape:
        temp_list.append(i)
    data_shape=tuple(temp_list)
    print(data_shape)

    #build dense model
    dense_model=tf.keras.Sequential()
    dense_model.add(tf.keras.Input(shape=data_shape))
    dense_model.add(tf.keras.layers.Dense(units=40, activation='tanh'))
    dense_model.add(tf.keras.layers.Dense(units=(20), activation='linear'))
    dense_model.add(tf.keras.layers.Dense(units=(20), activation='tanh'))
    dense_model.add(tf.keras.layers.Dense(units=(20), activation='linear'))
    dense_model.add(tf.keras.layers.Dense(units=(10), activation='tanh'))
    dense_model.add(tf.keras.layers.Dense(units=(1), activation='linear'))

    print(dense_model.summary())

    

    #fit model
    dense_model.compile(optimizer='adam', loss='mean_squared_error', metrics='mean_absolute_error')
    dense_model.fit(x=x_train, y=y_train, validation_split=0.3, shuffle=True, batch_size=16, epochs=50)

    #evaluate model
    print('Evaluating model')
    preds=dense_model.predict(x_test)
    plot_predictions(preds, y_test, name='regression'+str(LOOKBACK)+'_'+str(LOOKAHEAD)+'.png', save=False)
    plot_predictions(preds[-200:], y_test[-200:], name='regression'+str(LOOKBACK)+'_'+str(LOOKAHEAD)+'.png', save=False)
    print(dense_model.evaluate(x_test, y_test, batch_size=1))
    print('Accuracy:', direction_acc(y_test, preds))
    print('Baseline:', pd.Series(y_test).describe())
    return dense_model.evaluate(x_test, y_test, batch_size=1)