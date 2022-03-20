import numpy as np
import pandas as pd
from utilities import *
from evaluation import *

ticker = ["VALE3", "PETR4", "MGLU3", "ITUB4", "GOLL4"]

lookback = [5, 10, 20, 30, 50]
lookahead = [1, 5, 10, 15, 20]
test_err_arr = []
conditions = []

for lookback in lookback_arr:
    for lookahead in lookahead_arr:
        conditions.append((lookback, lookahead))
        print(conditions[-1])  # arr[-1] means the last element in the array
        test_err = run_lin_regression(ticker, lookback, lookahead)
        test_err_arr.append(test_err)

print(test_err_arr)
print(conditions)
