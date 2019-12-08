# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:42:59 2019

@author: vetur
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling,Linear Regression does it on its own

#Fitting simple linear onto the train sets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain,Ytrain)

#prediciting 
Ypred = regressor.predict(Xtest)

#graphs
plt.scatter(Xtest,Ytest,color = "red")
plt.plot(Xtest,Ypred,color ="blue")
plt.show()