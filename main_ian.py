import numpy as np
import pandas as pd
from utilities import *
from evaluation import *
from lin_regression import *
from svm import *

ticker = ["VALE3"]  # , "PETR4", "MGLU3", "ITUB4", "GOLL4"]

lookback_arr = [5]
lookahead_arr = [1]
# lookback_arr = [5, 10, 20]
# lookahead_arr = [1, 5, 10, 15, 20]
test_err_arr = []
conditions = []

for lookback in lookback_arr:
    for lookahead in lookahead_arr:
        conditions.append((lookback, lookahead))
        print(conditions[-1])  # arr[-1] means the last element in the array
        test_err = run_svm(ticker, lookback, lookahead)
        test_err_arr.append(test_err)

# print(test_err_arr)
print(conditions)
