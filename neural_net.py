import numpy as np
import pandas as pd
from utilities import *
import tensorflow as tf

TICKER='PETR4'
LOOKBACK=10
LABEL_COL='close'

#load data
raw_df=get_ticker_data(TICKER)

#get features
raw_df=create_features(raw_df, lookback=LOOKBACK, label_col=LABEL_COL)

#extract features
df=extract_features_2d(raw_df, lookback=LOOKBACK)
data_shape=np.shape(df.iloc[0]['data'])
print(df.iloc[0]['data'])
print(np.shape(df.iloc[0]['data']))

print(df['data'].isna().value_counts())
print(df['data'].dtypes)
#data_shape=(10,5,1)

#build model
model=tf.keras.Sequential()
model.add(tf.keras.Input(shape=data_shape))
model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(1,5)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=(10), activation='relu'))
model.add(tf.keras.layers.Dense(units=(1), activation='relu'))

print(model.summary())

print(df.iloc[0]['data'])
print(np.shape(df.iloc[0]['data']))

print(df['data'].isna().value_counts())
print(df['data'].dtypes)

#fit model
model.compile(optimizer='adam', loss='mean_squared_error', metrics='mean_squared_error')
model.fit(x=df['data'],y=df['label'], validation_split=0.3, shuffle=True, batch_size=1)

#evaluate model