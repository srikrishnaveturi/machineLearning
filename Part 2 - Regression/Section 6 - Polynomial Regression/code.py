# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:28:04 2019

@author: vetur
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
#we used 1:2 here coz we want x to be a matrix and not a 1d structure
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)""" 
#no splitting since there are already very few cases so well use all of them for training 

#making a linear regressor to compare it ith the polynomial regressor later
from sklearn.linear_model import LinearRegression
lReg = LinearRegression()
lReg.fit(X,y);

#now, the polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 3)
Xpoly = polyReg.fit_transform(X)
#we just transformed one row oof data into x,x^2,x^3 till x^n

#now we aremaking a multiple linear regression model with the values of Xpoly
lReg2 = LinearRegression()
lReg2.fit(Xpoly,y);

plt.scatter(X,y,color = 'red')
plt.plot(X,lReg.predict(X),color = 'blue')
plt.show()

plt.scatter(X,y,color = 'red')
plt.plot(X,lReg2.predict(polyReg.fit_transform(X)),color = 'blue')
plt.show()