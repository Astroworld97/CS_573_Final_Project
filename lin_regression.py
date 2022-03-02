from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

brazil_data_unclean = pd.read_csv(
    # "/Users/iangonzalez/Desktop/Spring_2022/CS_573_ML/CS_573_Final_Project/b3_stocks_1994_2020.csv"
    "b3_stocks_1994_2020.csv"
)

sns.heatmap(
    brazil_data_unclean.isnull(), yticklabels=False, cbar=False, cmap="cubehelix"
)
