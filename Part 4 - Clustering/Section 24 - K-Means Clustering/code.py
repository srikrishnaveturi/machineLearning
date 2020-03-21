# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:49:38 2020

@author: vetur
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values
#we used columns 3 and 4 because they are the columns of interest here(salary and spends)

#using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = list()
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("no. of clusters")
plt.ylabel("wcss")
plt.show()
#5 is found to be the elbow on the graph

#applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5,init = 'k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#visalising the clusters
#the following will only work for 2 columns of interest
plt.scatter(X[y_kmeans == 0][:,0],X[y_kmeans == 0][:,1], s = 50,c = 'red',label = 'careful')
plt.scatter(X[y_kmeans == 1][:,0],X[y_kmeans == 1][:,1], s = 50,c = 'blue',label = 'standard')
plt.scatter(X[y_kmeans == 2][:,0],X[y_kmeans == 2][:,1], s = 50,c = 'green',label = "target")
plt.scatter(X[y_kmeans == 3][:,0],X[y_kmeans == 3][:,1], s = 50,c = 'cyan',label = 'careless')
plt.scatter(X[y_kmeans == 4][:,0],X[y_kmeans == 4][:,1], s = 50,c = 'magenta',label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300,c = 'yellow',label = 'centroids')
plt.title("clusters")
plt.xlabel("salary")
plt.ylabel("spends")
plt.legend()
plt.show()