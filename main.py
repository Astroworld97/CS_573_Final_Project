from cProfile import run
import numpy as np
import pandas as pd
from utilities import *
import tensorflow as tf
from evaluation import *
from neural_net import *

#ticker=['VALE3', 'PETR4', 'MGLU3', 'ITUB4', 'GOLL4']
ticker=['VALE3', 'PETR4', 'MGLU3', 'VVAR3', 'ITUB4', 'GOLL4', 'LREN3', 'WEGE3', 'CTSA4', 'EZTC3', 'GGBR4',
            'MRFG3', 'SULA11', 'IRBR3', 'TAEE11', 'SUZB3']
#ticker=['VALE3', 'PETR4']

lookback_arr=[10]
lookahead_arr=[1]
test_err_arr=[]
conditions=[]


for lookback in lookback_arr:
    for lookahead in lookahead_arr:
        conditions.append((lookback, lookahead))
        print(conditions[-1])
        test_err = run_neural_net_regression(ticker, lookback, lookahead)
        test_err_arr.append(test_err)
        

print(test_err_arr)


#TODO:
#       - implement sma as added feature