# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 21:35:34 2020

@author: kaval
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('E:/Machine-Learning-master/DATA/Data_Ads.csv')

df.info()

df['Purchased'].value_counts()

sns.heatmap(df.isnull())

cnt = pd.get_dummies(df['Gender'],drop_first=True)
df = pd.concat([df,cnt],axis=1)

df.drop(['Gender'],axis=1,inplace=True)

x = df.drop(['Purchased'],axis=1).values

y= df['Purchased'].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

'''predicting the accuracy and confusion matix'''
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test,y_pred)
score = accuracy_score(y_test,y_pred)
