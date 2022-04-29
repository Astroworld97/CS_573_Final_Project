from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#function to create a dataframe with two columns: features(data) and labels
#each row of the data column has (lookback) raw data rows - meaning, we use the price of
#the previous ten days to predict the price of the eleventh day
def create_features(df, lookback=10, label_col='close', lookahead=1):
    df=df.sort_values('datetime')
    cols=['data','label', 'time', 'price', 'class', 'diff', 'ticker']
    feat_df=pd.DataFrame(columns=cols)
    for t in df['ticker'].unique():
        t_feat_df=pd.DataFrame(columns=cols)
        temp_df=df[df['ticker']==t]
        data=[]
        label=[]
        time=[]
        price=[]
        class_arr=[]
        diff=[]
        ticker=[]
        label.append(0)
        for i in range(lookback, len(temp_df)-lookahead):
            row=temp_df[i-lookback:i]
            data.append(row)
            temp_label=temp_df.iloc[i:i+lookahead][label_col].mean()
            if(temp_label>label[-1]):
                class_arr.append(1)
            else:
                class_arr.append(0)
            label.append(temp_label)
            time.append(temp_df.iloc[i]['datetime'])
            price.append(temp_df.iloc[i][label_col])
            diff.append(temp_label-temp_df.iloc[i-1][label_col])
            ticker.append(temp_df.iloc[i]['ticker'])
        label.remove(0)
        t_feat_df['data']=data
        t_feat_df['label']=label
        t_feat_df['time']=time
        t_feat_df['price']=price
        t_feat_df['class']=class_arr
        t_feat_df['diff']=diff
        t_feat_df['ticker']=ticker
        feat_df=pd.concat([feat_df, t_feat_df], ignore_index=True)
    return feat_df



#this function organizes the 'data' column into a 2d array, with 5 columns and 10 rows -deprecated :(
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

#this function organizes the 'data' columns into a 1d array, with length 5*10 -deprecated lol
def extract_features_1d(df, lookback=10):
    data=[]
    cols_of_interest=['open', 'high', 'low', 'close', 'volumen']
    for i in range(len(df)):
        row_df=df.iloc[i]['data']
        data_i=[]
        for j in range(lookback):
            for col in cols_of_interest:
                data_i.append(row_df.iloc[j][col])
                assert type(row_df.iloc[j][col])==np.float64
        data.append(np.array(data_i, dtype=np.float64).reshape(1,len(cols_of_interest)*lookback))
        data[i]=data[i][0]
        
    new_df=pd.DataFrame(columns=['data', 'label'])
    print('type:', pd.Series(data).dtypes)
    print('type instance 0:', type(data[0]))
    np_data=np.array(data).astype(np.float32)
    print('type:', type(np_data))
    print('type instance 0:', type(np_data[0]))
    print('type instance 0, element 0:', type(np_data[0][0]))
    new_df['data']=np_data
    new_df['label']=df['label']
    return new_df

def extract_features_cols(df, lookback=10):
    cols_of_interest=['open', 'high', 'low', 'close', 'volumen']
    out_df=pd.DataFrame()
    for i in range(1,lookback+1):
        for col in cols_of_interest:
            col_name=str(i)+'_'+col
            temp_arr=[]
            for k in range(len(df)):
                temp_arr.append(df.iloc[k]['data'].iloc[i-1][col])
            out_df[col_name]=temp_arr
    #add label col
    out_df.loc[:,'label']=df['label']
    #add time col
    out_df.loc[:,'time']=df['time']
    #add price col
    out_df.loc[:,'price']=df['price']
    #add class col
    out_df.loc[:,'class']=df['class']
    #add class col
    out_df.loc[:,'diff']=df['diff']
    #add ticker col
    out_df['ticker']=df['ticker']
    return out_df


def get_ticker_data(ticker_arr):
    out_df=pd.DataFrame()
    df=pd.read_csv('b3_stocks_1994_2020.csv')
    for ticker in ticker_arr:
        temp_df=df[df['ticker']==ticker]
        out_df=pd.concat([out_df, temp_df])
    return out_df

def get_sma(arr: pd.Series, period):
    sma=[]
    for i in range(period, len(arr)):
        sma.append(arr[i-period-1:i-1].mean())
    return sma

def split_test_set(df: pd.DataFrame, by_ticker=False, test_ticker='PETR4'):
    test_df=pd.DataFrame()
    train_df=pd.DataFrame()
    if(not by_ticker):
        test_df=df[df['time']>'2020-01-01']
        train_df=df[df['time']<'2020-01-01']
    else:
        test_df=df[df['ticker']==test_ticker].copy(deep=True).reset_index(drop=True)
        train_df=df[df['ticker']!=test_ticker].copy(deep=True).reset_index(drop=True)
    return train_df, test_df

def normalize_cols(df, cols): #min max norm
    for col in cols:
        df.loc[:, col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())

def standardize_cols(df, cols):
    for col in cols:
        df.loc[:, col]=(df[col]-df[col].mean())/df[col].std()

#TODO
def split_test_set_by_ticker():
    pass