# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:47:53 2020

@author: vetur
"""

#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

#using dendrograms for optimal numbers of clusters

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("dendrogram")
plt.xlabel("customers")
plt.ylabel("euclidian distances")
plt.show()

#fitting hierarchical clustering to the mall dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity = "euclidean",linkage = "ward")
y_hc = hc.fit_predict(X)

#visualising the clusters
#the following will only work for 2 columns of interest
plt.scatter(X[y_hc == 0][:,0],X[y_hc == 0][:,1], s = 50,c = 'red',label = 'careful')
plt.scatter(X[y_hc == 1][:,0],X[y_hc == 1][:,1], s = 50,c = 'blue',label = 'standard')
plt.scatter(X[y_hc == 2][:,0],X[y_hc == 2][:,1], s = 50,c = 'green',label = "target")
plt.scatter(X[y_hc == 3][:,0],X[y_hc == 3][:,1], s = 50,c = 'cyan',label = 'careless')
plt.scatter(X[y_hc == 4][:,0],X[y_hc == 4][:,1], s = 50,c = 'magenta',label = 'sensible')
plt.title("clusters")
plt.xlabel("salary")
plt.ylabel("spends")
plt.legend()
plt.show()