import numpy as np
import pandas as pd
import sys
import math

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection  import train_test_split 

df = pd.read_csv("data/tokyodata.csv")
scalar = StandardScaler()
X = df[["year", "month", "day", "最高気温", "最低気温", "日照時間"]]
y = df[["降水量"]]

X["year"]  = np.sin(X["year"])
X["month"] = np.sin(X["month"])
X["day"]   = np.sin(X["day"])
X["最高気温"] = scalar.fit_transform(X["最高気温"])
X["最低気温"] = scalar.fit_transform(X["最低気温"])
X["日照時間"] = scalar.fit_transform(X["日照時間"])

print(X)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=0)

alpha = np.average(X["日照時間"])
reg = MLPRegressor(X_train,
                   y_train,
                   alpha=alpha)

year = int(sys.argv[1])
month = int(sys.argv[2])
day = int(sys.argv[3])
max_temperature = float(sys.argv[4])
min_temperature = float(sys.argv[5])

# 説明変数の正規化
year_sin  = math.sin(year)
month_sin = math.sin(month)
day_sin   = math.sin(day)
max_temperature_sin = math.sin(max_temperature)
min_temperature_sin = math.sin(min_temperature)


predict_data = [year_sin,
                month_sin,
                day_sin,
                max_temperature_sin,
                min_temperature_sin]

result = reg.predict(predict_data)
result = result[0]

print()
print(f"{year}年{month}月{day}日の降水量は{result}と予測されました。")