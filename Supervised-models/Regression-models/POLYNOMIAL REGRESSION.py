# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:49:42 2020

@author: kaval
"""
'''AS THE DATASET IS SMALL WE USING COMPLETE DATA FOR TRAINING'''

'''Importing the libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''Importing the dataset'''
dataset = pd.read_csv('DATA/Position_Salaries.csv')

'''Droping the position'''
dataset.drop(['Position'],axis=1,inplace=True)

''' Extracting features & target '''
x = dataset.drop(['Salary'],axis=1).values
y = dataset['Salary'].values

'''Visualizing the data'''
plt.scatter(x,y)

'''Importing the Linear model'''
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
lin_pred = model.predict(x)

'''Importing the polynomial model'''
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree=3)
x_poly = polyreg.fit_transform(x)
model2 = LinearRegression()
model2.fit(x_poly,y)
poly_pred = model2.predict(x_poly)

'''Visualising the Linear Regression results'''
plt.scatter(x, y, color = 'red')
plt.plot(x,lin_pred, color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''Visualising the Polynomial Regression results'''
plt.scatter(x, y, color = 'red')
plt.plot(x, poly_pred, color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''Smoothing the regression line of polynomial model'''
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, model2.predict(polyreg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''Predicting a new result with Linear Regression'''
print(model.predict([[6.5]]))

'''Predicting a new result with Polynomial Regression'''
print(model2.predict(polyreg.fit_transform([[6.5]])))

'''Accuracy of linear model'''
from sklearn.metrics import r2_score
score1 = r2_score(y,lin_pred)

'''Accuracy of polynomial model'''
score2 = r2_score(y,poly_pred)