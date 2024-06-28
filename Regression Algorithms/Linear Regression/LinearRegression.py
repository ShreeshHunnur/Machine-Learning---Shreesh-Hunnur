import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Salary_Data.csv")
# print(data.head())
# print(data.isnull().sum())

#DATA VISUALIZATION
# plt.scatter(data["YearsExperience"],data["Salary"])
# plt.xlabel("Experience in Years")
# plt.ylabel("Salary")
# plt.title("Years of Experience vs Salary")
# plt.show()

# DATA SPLITTING in X and Y
y = pd.DataFrame(data["Salary"])
# X = data.iloc[:,0:1]
X = pd.DataFrame(data["YearsExperience"])

# print(X)
# print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# print(X_train)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# MODEL BUILDING - LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# TRAINING THE MODEL
lr.fit(X_train,y_train)

# PREDICTION (TEST THE MODEL)
y_pred = lr.predict(X_test)
# print(y_pred)
# print(y_test)

#EVALUATING MODEL(checking accuracy)
from sklearn.metrics import r2_score
acc = r2_score(y_pred,y_test)
# print(acc)

#PREDICTING FOR SOME RANDOM VALUES

pred_random = lr.predict([[40]])
# print(pred_random)

# plt.scatter(X_train,y_train)
# plt.plot(X_train,lr.predict(X_train),"r")
# plt.show()

# Getting SLOPE AND INTERCEPT ie M and C

print(lr.coef_)
print(lr.intercept_)
