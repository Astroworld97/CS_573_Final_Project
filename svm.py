from sklearn.svm import SVR
from sklearn.svm import LinearSVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math

brazil_data = pd.read_csv("b3_stocks_1994_2020.csv")

# start open
data = brazil_data.loc[brazil_data["ticker"] == "PETR4"]
data = data.reset_index()
features = ["close", "high", "low", "volumen"]
new_data = pd.DataFrame(index=range(0, len(data)), columns=features)
new_data["close"][0] = 0
new_data["high"][0] = 0
new_data["low"][0] = 0
new_data["volumen"][0] = 0
for i in range(1, len(data)):
    new_data["close"][i] = data["close"][i - 1]
    new_data["high"][i] = data["high"][i - 1]
    new_data["low"][i] = data["low"][i - 1]
    new_data["volumen"][i] = data["volumen"][i - 1]

x = new_data[features]
y = data["open"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svr_rbf_results = []
for i in range(900, 1000):
    svr_rbf = SVR(kernel="rbf", C=i)
    svr_rbf.fit(x_train, y_train)
    svr_rbf_confidence = svr_rbf.score(x_test, y_test)
    print(
        "svr_rbf confidence, aka R^2: "
        + str(svr_rbf_confidence)
        + "for a C parameter of: "
        + str(i)
    )


# Note: problems with both SVR(kernel="linear") and LinearSVR()
# svr_lin = LinearSVR(penalty="l1", loss="l2", dual=False)
# svr_lin.fit(x_train, y_train)
# svr_lin = SVR(kernel="linear", C=1)
# svr_lin.fit(x_train, y_train)

# svr_poly = SVR(kernel="poly")
# svr_poly.fit(x_train, y_train)

# for i in range(1, 1000):
#     svr_poly = SVR(kernel="poly", C=i)
#     svr_poly.fit(x_train, y_train)
#     svr_poly_confidence = svr_poly.score(x_test, y_test)
#     print(
#         "svr_poly confidence, aka R^2: "
#         + str(svr_poly_confidence)
#         + "for a C parameter of: "
#         + str(i)
#     )

# svr_rbf_confidence = svr_rbf.score(x_test, y_test)
# print("svr_rbf confidence, aka R^2: " + str(svr_rbf_confidence))

# svr_lin_confidence = svr_lin.score(x_test, y_test)
# print("svr_lin confidence: " + svr_lin_confidence)

# svr_poly_confidence = svr_poly.score(x_test, y_test)
# print("svr_poly confidence, aka R^2: " + str(svr_poly_confidence))
