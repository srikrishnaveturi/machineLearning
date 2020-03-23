# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:34:32 2020

@author: vetur
"""
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv",header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])

#training apriori on the dataset
#you'll need the two python attached files,apyori.py and apriori,py
from apyori import apriori
rules = apriori(transactions, min_support = 0.003,min_confidence = 0.2,min_lift = 3,min_length = 2)

#visualizing the results
results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nInfo:\t' + str(results[i][2]))