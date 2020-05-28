 # -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:37:15 2020

@author: vetur
"""

#preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('Churn_Modelling.xlsx')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print(X.shape)

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([("Geography ", OneHotEncoder(),[1])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
X = ct.fit_transform(X)
for a in X[0,:]:
    print(a)
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
#create and fit your model here

#importing the keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()

#adding the  input layer and the first hidden layer
classifier.add(Dense(units=6, input_dim = 11, activation='relu', kernel_initializer="uniform"))

#adding the second hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer="uniform"))
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

#compiling the ANN(training)

classifier.compile(optimizer = "adam", loss = "binary_crossentropy")
#the loss function is the value that will be optimised(mostly minimised) by the optimiser

#fitting the classifier with X_train and y_train
classifier.fit(X_train,y_train,batch_size = 20,epochs = 50)
#making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_test = (y_test > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
