# We have 2 or more independent variables and 1 dependent variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# DATA LOADING
data = pd.read_csv("50_Startups.csv")
# print(data.isnull().sum())

# print(data)
# print(data["State"].value_counts())


# ENCODING STATE variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data.State = le.fit_transform(data.State)
# print(data)

# SPLITTING X AND Y

y = pd.DataFrame(data["Profit"])
x = data.drop(columns=["Profit"], axis = 1)

# print(x)
# print(y)

# SCALING INDEPENDENT VARIABLES using STANDARD SCALER

from sklearn.preprocessing import StandardScaler

names = x.columns

scale = StandardScaler()

x = scale.fit_transform(x)

x = pd.DataFrame(x,columns=names)
# print(x)

# TEST TRAIN SPLITTING

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_test,y_test)

pred = model.predict(x_test)

from sklearn.metrics import r2_score

acc = r2_score(pred,y_test)
print("Accuracy: ",acc)

print(model.coef_)
print(model.intercept_)
print("RANDOM Prediction: ",model.predict([[40,60,30,0]]))

