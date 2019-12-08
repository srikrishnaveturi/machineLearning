#datapreprocessing

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataSet = pd.read_csv("Data.csv")
x = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,3].values

#missingdata
#mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN",strategy = "mean",axis = 0)
x[:,1:3] = imputer.fit_transform(x[:,1:3])


#encoding the catagories
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoderx  = LabelEncoder()
x[:,0] = labelencoderx.fit_transform(x[:,0])
ohe = OneHotEncoder(categorical_features = [0])
x = ohe.fit_transform(x).toarray()
labelencodery = LabelEncoder()
y = labelencodery.fit_transform(y)"""

#splitting the data set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size = 0.2,random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
xTrain = scx.fit_transform(xTrain)
xTest = scx.transform(xTest)"""
