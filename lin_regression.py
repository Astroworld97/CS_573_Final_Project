from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
from utilities import *


def run_lin_regression(TICKER, LOOKBACK, LOOKAHEAD, LABEL_COL="close"):

    raw_df = get_ticker_data(TICKER)
    raw_df = create_features(raw_df, lookback=LOOKBACK, label_col=LABEL_COL)
    temp = extract_features_cols(raw_df, lookback=LOOKBACK)
    return temp
