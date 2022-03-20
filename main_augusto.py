from cProfile import run
import numpy as np
import pandas as pd
from utilities import *
import tensorflow as tf
from evaluation import *
from neural_net import *

ticker = ["VALE3", "PETR4", "MGLU3", "ITUB4", "GOLL4"]

lookback_arr = [10, 20, 50]
lookahead_arr = [1, 5, 20]
test_err_arr = []
conditions = []


for lookback in lookback_arr:
    for lookahead in lookahead_arr:
        conditions.append((lookback, lookahead))
        print(conditions[-1])
        test_err = run_neural_net(ticker, lookback, lookahead)
        test_err_arr.append(test_err)


print(test_err_arr)
print(conditions)
