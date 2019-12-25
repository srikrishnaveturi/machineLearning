# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:37:50 2019

@author: vetur
"""

# this type of model breaks the data set into leaves and predicts the average of the results for any independent variable that might lie in the same leaf

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y= sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
# Create your regressor here

# Predicting a new result
y_pred = regressor.predict([[6.5]])

# Visualising the decsion tree results
#we're using this type of a plot because we want to see thst the model predicted a single balue for each "leaf"
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (decision tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
