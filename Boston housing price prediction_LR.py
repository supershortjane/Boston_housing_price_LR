#!/usr/bin/env python
# coding: utf-8

# # Linear Regression  
# dataset description: https://www.kaggle.com/c/boston-housing  

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
#from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing


# In[3]:


df=pd.read_csv('housing.csv',header=None,delim_whitespace=True)


# In[4]:


df.head()


# In[5]:


df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df


# In[6]:


#to see the distribution of each column
df.describe()


# In[7]:


#there is no null values in this dataset
#the dataset has 14 columns and 506 instances
df.info()


# In[10]:


y = df['MEDV']
X = df.drop(['MEDV'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients: {}\n'.format(model.coef_))
# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 
print('R2 score: {}'.format(r2_score(y_test, y_pred)))


# In[11]:


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual V.S. Predicted')


# In[ ]:




