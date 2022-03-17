from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")

# print(brazil_data.loc[:, "open"])

ace_data = brazil_data.loc[brazil_data["ticker"] == "ACE 3"]

vale_data = brazil_data.loc[brazil_data["ticker"] == "VALE3"]
print(vale_data.head())
print(vale_data.tail())
