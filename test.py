# test
# website of data: https://www.kaggle.com/felsal/ibovespa-stocks?select=b3_stocks_1994_2020.csv
import pandas as pd
from utilities import *
from lin_regression import *

brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")
ticker = ["VALE3", "PETR4", "MGLU3", "ITUB4", "GOLL4"]
x = run_lin_regression(ticker[0], 10, 5)
print(x)
