import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# DATA COLLECTION

data = pd.read_csv("Position_Salaries.csv")
# print(data.head())

# print(data.isnull().sum())

#FOR NOW SPLITTING DIRECTLY

x = data.iloc[:,1:2]
y = data.iloc[:,2:]

# plt.scatter(x,y)
# plt.show()

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=5)

x_poly = pr.fit_transform(x)
x_poly

pr.fit(x_poly, y)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

model = lr.fit(x_poly,y)

pred = model.predict(x_poly)

plt.scatter(x,y)
plt.plot(x,pred,"r")
plt.show()