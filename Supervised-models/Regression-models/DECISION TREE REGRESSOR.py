# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 03:53:05 2020

@author: kaval
"""

'''Importing the libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''Importing the dataset'''
dataset = pd.read_csv('DATA\Clustering - Assignment-1.csv')

'''Visualize the data'''
sns.pairplot(dataset,x_vars=['AT','V','AP','RH'],y_vars=['PE'])

'''checking any null values'''
sns.heatmap(dataset.isnull(),cmap='plasma')

'''Extracting the Features'''
x = dataset.drop('PE',axis=1).values
y = dataset['PE'].values

''' Spliting the dataset into training and testing '''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

'''Scaling the values'''
from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
x_train[:,:] = scaling.fit_transform(x_train[:,:])
x_test[:,:] = scaling.transform(x_test[:,:])

'''Importing the DecisionTreeRegressor model'''
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)

'''Traning the model'''
model.fit(x_train,y_train)

'''Predicting the results'''
y_pred = model.predict(x_test)

'''Predicting the accuracy of DecisionTreeRegressor model'''
from sklearn.metrics import r2_score
score = r2_score(y_pred,y_test)

