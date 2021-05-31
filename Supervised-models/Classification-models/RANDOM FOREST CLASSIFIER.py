# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 00:42:57 2020

@author: kaval
"""
'''Importing the libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''Importing the dataset'''
data = pd.read_csv('DATA\Classification Assignment - Loan_Application.csv')

'''gathering information about the data'''
data.info()
data.describe()

'''Visualizing the missing values'''
sns.heatmap(data.isnull(),cmap='plasma')

'''Encoding the Independent Variable'''
cmt = pd.get_dummies(data['purpose'],drop_first=True)
data = pd.concat([data,cmt],axis=1)

'''updating the dataset'''
data.drop(['purpose'],axis=1,inplace=True)

'''Extracting the features and labels'''
x = data.drop(['not.fully.paid'],axis=1).values
y = data['not.fully.paid'].values

''' Spliting the dataset into training and testing '''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=10)

'''Importing the logistic regression model'''
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

'''Training the data'''
log.fit(x_train,y_train)

'''Predicting the results'''
y_pred = log.predict(x_test)

'''predicting the accuracy and confusion matix'''
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test,y_pred)
score = accuracy_score(y_test,y_pred)

'''Importing the DecisionTreeClassifier model'''
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',random_state=0)

'''Training the data'''
tree.fit(x_train,y_train)

'''Predicting the results'''
y_pred2 = tree.predict(x_test)

'''predicting the accuracy and confusion matix'''
confusion_matrix(y_test,y_pred2)
score2 = accuracy_score(y_test,y_pred2)

'''Importing the RandomForestClassifier model'''
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)

'''Training the data'''
forest.fit(x_train,y_train)

'''Predicting the results'''
y_pred3 = forest.predict(x_test)

'''predicting the accuracy and confusion matix'''
confusion_matrix(y_test,y_pred3)
score3 = accuracy_score(y_test,y_pred3)

plt.bar(['Logistic','Decision','Random'],[score,score2,score3])
plt.title('Comparsion of Classification models')
plt.ylabel('accuracy')
