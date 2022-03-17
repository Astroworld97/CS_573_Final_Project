# test
# website of data: https://www.kaggle.com/felsal/ibovespa-stocks?select=b3_stocks_1994_2020.csv
import pandas as pd

brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")

test = brazil_data.loc[brazil_data["ticker"] == "CTB 4"]

print(test.head())
