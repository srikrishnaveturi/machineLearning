# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:21:27 2020

@author: vetur
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t",quoting = 3)

#cleaning the texts in the dataset
import re
import nltk
nltk.download("stopwords") #no need to run this line if already downloaded
from nltk.corpus import stopwords
#stopwords are words used to form the sentence but have no relation to the emotion of the sentence
from nltk.stem.porter import PorterStemmer
#this class is used for "stemming" which means assigning the same word for words which are the synonyms,almost like "love", "loved" and "will love" all will be given out as "love"
corpus = []
for i in range(1000):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    #these lines excluded all the characters and numbers except letters and made all the words lower caseand added them into a list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    #we included set() because it is faster to searh in set than a list, so if the list is huge,typecast it into a set and then search
    review = (" ").join(review)
    corpus.append(review)

#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
#we'll only keep 1500 most frquent words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Logistic Regression to the Training set
#create and fit your model here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)