import numpy as np
import pandas as pd
import sys
import math

from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection  import train_test_split 

df = pd.read_csv("src/data/tokyodata.csv")
scalar = StandardScaler()
X = df[["year", "month", "day", "最高気温", "最低気温", "日照時間"]]
y = df[["降水量"]]

X["year"]  = np.sin(X["year"])
X["month"] = np.sin(X["month"])
X["day"]   = np.sin(X["day"])

alpha = df[["日照時間", "全天日射量", "平均気温", "平均雲量"]]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=0)

alpha = np.average((alpha["日照時間"] + alpha["全天日射量"]) * (alpha["平均気温"] + alpha["平均雲量"]))
reg = MLPRegressor(alpha =alpha,
                   verbose=True,
                   random_state=0)

year  = int(sys.argv[1])
month = int(sys.argv[2])
day   = int(sys.argv[3])
max_temperature = float(sys.argv[4])
min_temperature = float(sys.argv[5])
solar_time      = float(sys.argv[6])
# 日付の正規化
year_sin  = math.sin(year)
month_sin = math.sin(month)
day_sin   = math.sin(day)


predict_data = [[year_sin,
                month_sin,
                day_sin,
                max_temperature,
                min_temperature,
                solar_time]]

reg.fit(X_train, y_train)
result = reg.predict(predict_data)
result = result[0]

score_train  = reg.score(X_train, y_train)
score_test   = reg.score(X_test, y_test)
print()
print(f"Model Score (Train) : {score_train}")
print(f"Model Score (Test)  : {score_test}")
print()
print(f"{year}年{month}月{day}日の降水量は{result}[mm]と予測されました。")