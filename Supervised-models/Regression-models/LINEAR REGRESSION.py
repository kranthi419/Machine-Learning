# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 03:27:21 2020

@author: kaval
"""
'''Importing the libraries '''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

''' Importing the Dataset '''

dataset=pd.read_csv('DATA/Salary_Data.csv')

''' Extracting the features '''

x=dataset.drop(['Salary'],axis=1).values
y=dataset['Salary'].values 

''' Ploting the relation between features and target '''

plt.scatter(x,y)
plt.xlabel('Experience')
plt.ylabel('Salary')

''' Spliting the dataset into training and testing '''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

''' Importing the model '''

from sklearn.linear_model import LinearRegression
model=LinearRegression()

''' Fitting the data or giving the data to model '''

model.fit(x_train,y_train)

''' Predicting the values '''

y_predict=model.predict(x_test)

''' Ploting the regression line '''

plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_predict, color='red', linewidth=2)

'''Plotting the bar graph for actual and predicted values'''

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predict.flatten()})
df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.ylabel('Salaries')
