# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 06:26:45 2020

@author: kaval
"""
'''AS THE DATASET IS SMALL WE USING COMPLETE DATA FOR TRAINING'''

'''Importing the libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing of dataset'''
dataset = pd.read_csv('DATA\Position_Salaries.csv')

'''droping the columns which is not necessary'''
dataset.drop(['Position'],axis=1,inplace=True)

'''Extracting the features'''
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,1].values

'''Visualize the data'''
plt.scatter(x,y,color='b')
plt.xlabel('level')
plt.ylabel('Salary')
plt.title('POSITION_SALARIES')

'''Importing the model'''
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=30)

'''Training the model'''
model.fit(x,y)

'''predicting the results'''
y_pred = model.predict(x)

'''Visualize the results'''
plt.scatter(x,y,color='b')
plt.plot(x,y_pred,linewidth=2,color='r')
plt.xlabel('level')
plt.ylabel('Salary')
plt.title('POSITION_SALARIES')

df = pd.DataFrame({'Actual': y, 'Predicted':y_pred})
df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

'''Finding the accuracy'''
from sklearn.metrics import r2_score
score = r2_score(y,y_pred)