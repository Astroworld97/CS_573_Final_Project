from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math

brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")

vale_data = brazil_data.loc[brazil_data["ticker"] == "VALE3"]
