# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:46:34 2020

@author: kaval
"""

''' Importing the libraries '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

''' Importing the dataset '''

dataset = pd.read_csv('DATA/Data.csv')

''' Encoding the categorical data '''
''' Encoding the Independent Variable '''

cnt = pd.get_dummies(dataset['Country'],drop_first=True)
dataset=pd.concat([dataset,cnt],axis=1)

''' Encoding the dependent Variable '''

lb = pd.get_dummies(dataset['Purchased'],drop_first=True)
dataset = pd.concat([dataset,lb],axis=1)

''' update dataset '''

dataset.drop(['Country'],axis=1,inplace=True)
dataset.drop(['Purchased'],axis=1,inplace=True)

''' Extracting features & target '''

x = dataset.drop(['Yes'],axis=1).values
y = dataset['Yes'].values

''' Handling the missing data '''

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,0:2])
x[:,0:2]=imputer.transform(x[:,0:2])

''' Spliting the dataset into training and testing '''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

''' Scaling the values '''

from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
x_train[:,0:2]=scaling.fit_transform(x_train[:,0:2])
x_test[:,0:2]=scaling.transform(x_test[:,0:2])





