#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the necessary libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[8]:


#Upload the file "insurance.csv"
file = pd.read_csv("insurance.csv")

#relabeling the "sex" and "smoker" columns to 0 and 1
file['smoker'] = file['smoker'].map({'yes': 1, 'no' : 0})
file['sex'] = file['sex'].map({'male': 1, 'female' : 0})


# In[16]:


#assigning the variable columns to X and the dependent variable "charges" to Y
X = file[['age', 'bmi', 'children','smoker','sex']]
Y = file['charges']


# In[ ]:


#create some features to the degree "2"
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


# In[18]:


#scale the features
scale = StandardScaler()
scale.fit(X_poly)
X_scale = scale.transform(X_poly)


# In[19]:


#Spliting the data set into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size = 0.2, random_state = 0)


# In[20]:


#instatiation of the linear model object
lm = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# In[21]:


#traing the model
lm.fit(X_train, Y_train)


# In[22]:


#test the model
Y_pred = lm.predict(X_test)


# In[ ]:


#some performance indicators
print("intercept: \n",lm.intercept_)
print("coefficients: \n", lm.coef_)
print("absolute error:\n", mean_absolute_error(Y_test, Y_pred))

