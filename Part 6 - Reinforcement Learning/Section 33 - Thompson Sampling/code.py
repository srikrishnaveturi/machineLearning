# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:08:38 2020

@author: vetur
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#implementing the thompson sampling
import random
N = 10000
d = 10
adsSelected = []
NumberOfRewards_1 = [0] * d
NumberOfRewards_0 = [0] * d
totalRewards = 0
for n in range(N):
    maxRandom = 0
    ad = 0
    #for the staring 10 values of n, each ad will be selected once and that is why we added the if condition
    for i in range(d):
       randomBeta = random.betavariate(NumberOfRewards_1[i] + 1,NumberOfRewards_0[i] + 1)
       if maxRandom < randomBeta:
           maxRandom = randomBeta
           ad = i;
    adsSelected.append(ad)
    if dataset.values[n,ad]:
        NumberOfRewards_1[ad] += 1
    else:
        NumberOfRewards_0[ad] += 1
    totalRewards += dataset.values[n,ad]
    
#visualising the results
plt.hist(adsSelected)
plt.title("histogram of ads selections")
plt.xlabel("ad number")
plt.ylabel("no. of times ads were selected")
plt.show()
