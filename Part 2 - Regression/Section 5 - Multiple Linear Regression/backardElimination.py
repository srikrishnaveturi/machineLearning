# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:24:38 2019

@author: vetur
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#categorical data is present in the form of country

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#removing one dummy variable
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

Pred = regressor.predict(X_test)

#building a model using backward elimination
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50, 1), dtype = int), values = X, axis = 1)
Xopt = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = Xopt).fit()
regressor_ols.summary()
Xopt = X [:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = Xopt).fit()
regressor_ols.summary()
Xopt = Xopt[:,[0,2,3,4]]
regressor_ols = sm.OLS(endog = y,exog = Xopt).fit()
regressor_ols.summary()
Xopt = Xopt[:,[0,1,3]]
regressor_ols = sm.OLS(endog = y,exog = Xopt).fit()
regressor_ols.summary()
Xopt = Xopt[:,[0,1]]
regressor_ols = sm.OLS(endog = y,exog = Xopt).fit()
regressor_ols.summary()




