# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:30:13 2020

@author: vetur
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#implementing upper confidence bound algorithm
import math
N = 10000 #number of rows
d = 10 #number of ads
adsSelected = []
numberOfSelections = [0] * d
sumsOfRewards =  [0] * d
totalRewards = 0
for n in range(N):
    maxUpperBound = 0
    ad = 0
    #for the staring 10 values of n, each ad will be selected once and that is why we added the if condition
    for i in range(d):
        if numberOfSelections[i] > 0:
            averageReward = sumsOfRewards[i] / numberOfSelections[i]
            delta_i = math.sqrt(1.5 * math.log(n + 1) / numberOfSelections[i])
            #n+1  coz n starts with 0 but the formula expects n from 1
            upperBound = averageReward + delta_i
        else:
            upperBound = 1e400 #just to have a super large value
        if maxUpperBound < upperBound:
            maxUpperBound = upperBound
            ad = i;
    adsSelected.append(ad)
    numberOfSelections[ad]+=1
    sumsOfRewards[ad] += dataset.values[n,ad]
    totalRewards += dataset.values[n,ad]
    
#visualising the results
plt.hist(adsSelected)
plt.title("histogram of ads selections")
plt.xlabel("ad number")
plt.ylabel("no. of times ads were selected")
plt.show()
