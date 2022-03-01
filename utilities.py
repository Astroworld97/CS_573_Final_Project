from dataclasses import dataclass
import pandas as pd
import numpy as np

def create_features(df, lookback=10, label_col='close'):
    df=df.sort_values('datetime')
    cols=['data','label']
    feat_df=pd.DataFrame(columns=cols)
    data=[]
    label=[]
    for i in range(lookback, len(df)):
        row=df[i-lookback:i]
        data.append(row)
        label.append(df.iloc[i][label_col])
    feat_df['data']=data
    feat_df['label']=label
    return feat_df

def extract_features_2d(df, lookback=10):
    data=[]
    cols_of_interest=['open', 'high', 'low', 'close', 'volumen']
    for i in range(len(df)):
        row_df=df.iloc[i]['data']
        data_i=[]
        for j in range(lookback):
            row=[]
            for col in cols_of_interest:
                row.append(row_df.iloc[j][col])
            assert np.shape(row)==(5,)
            data_i.append(np.array(row, dtype=np.float32))
        assert np.shape(data_i)==(10,5)
        data.append(np.array(data_i, dtype=np.float32).reshape(lookback,len(cols_of_interest),1))
    new_df=pd.DataFrame(columns=['data', 'label'])
    new_df['data']=data
    new_df['label']=df['label']
    return new_df

def extract_features_1d(df, lookback=10):
    data=[]
    cols_of_interest=['open', 'high', 'low', 'close', 'volumen']
    for i in range(len(df)):
        row_df=df.iloc[i]['data']
        data_i=[]
        for j in range(lookback):
            for col in cols_of_interest:
                data_i.append(row_df.iloc[j][col])
        data.append(np.array(data_i, dtype=np.float64))
    new_df=pd.DataFrame(columns=['data', 'label'])
    new_df['data']=data
    new_df['label']=df['label']
    return new_df

def get_ticker_data(ticker):
    df=pd.read_csv('b3_stocks_1994_2020.csv')
    return df[df['ticker']==ticker]

def get_sma(period):
    pass