# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:02:01 2020

@author: kaval
"""
''' Importing the libraries ''' 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

''' Importing the dataset '''

dataset = pd.read_csv('DATA/Companies.csv')

''' Encoding the categorical data '''
''' Encoding the Independent Variable '''

cnt = pd.get_dummies(dataset['State'],drop_first=True)
dataset = pd.concat([dataset,cnt],axis=1)

''' update dataset '''

dataset.drop(['State'],axis=1,inplace=True)

''' Extracting features & labels '''

x = dataset.drop('Profit',axis=1).values
y = dataset['Profit'].values

''' Spliting the data into training and testing '''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

''' Scaling the values '''

from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
x_train[:,0:3] = scaling.fit_transform(x_train[:,0:3])
x_test[:,0:3] = scaling.transform(x_test[:,0:3])

''' Importing the model '''

from sklearn.linear_model import LinearRegression
model = LinearRegression()

''' Fitting the data or giving the data to model '''

model.fit(x_train,y_train)

''' Predicting the values '''

y_pred = model.predict(x_test)

''' Plotting the bar graph for actual and predicted values '''

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.ylabel('profits')

''' predicting the score of our model '''

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)


