# test
# website of data: https://www.kaggle.com/felsal/ibovespa-stocks?select=b3_stocks_1994_2020.csv
import pandas as pd
from utilities import *
from pylab import rcParams
import matplotlib.pyplot as plt


brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")  # returns a dataframe

tickers = ["VALE3", "PETR4", "MGLU3", "ITUB4", "GOLL4"]
ticker = [tickers[0]]  # "VALE3"
raw_df = get_ticker_data(ticker)
print(raw_df)

""" The comments below print the original 'close' for "VALE3"
rcParams["figure.figsize"] = 10, 8  # width 10, height 8

ax = raw_df.plot(x="datetime", y="close", style="b-", grid=True)
ax.set_xlabel("date")
ax.set_ylabel("USD")
plt.show()
"""

train_size = 0.7
validation_size = 0.2
test_size = 0.1

# RMSE = []
# mape = []
# for N in range(1, Nmax+1): # N is no. of samples to use to predict the next value
#     est_list = get_preds_mov_avg(train_cv, 'adj_close', N, 0, num_train)

#     cv.loc[:, 'est' + '_N' + str(N)] = est_list
#     RMSE.append(math.sqrt(mean_squared_error(est_list, cv['adj_close'])))
#     mape.append(get_mape(cv['adj_close'], est_list))
# print('RMSE = ' + str(RMSE))
# print('MAPE = ' + str(mape))

# raw_df = create_features(raw_df, lookback=10, label_col="close")
# temp = extract_features_cols(raw_df, lookback=10)
# print(type(temp))
# N1 = len(temp) / 3
# N2 = 2 * len(temp) / 3
# N3 = len(temp)
# min_val = temp["close"].min()
# mov_avg1 = get_preds_mov_avg(temp, "close", N1, min_val, 0) #offset set to zero so it provides predictions for the whole dataset
# mov_avg2 = get_preds_mov_avg(temp, "close", N2, min_val, 0)
# mov_avg3 = get_preds_mov_avg(temp, "close", N3, min_val, 0)
