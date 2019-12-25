# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 17:01:10 2019

@author: vetur
"""
#this type of regression uses multiple decision trees with different randomly pcked data and give the average of all the results of all the trees to give a better and more accurate model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0,)
regressor.fit(X,y)
# Create your regressor here

# Predicting a new result
y_pred = regressor.predict([[6.5]])

#visualizing the Random forest regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random forest regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
