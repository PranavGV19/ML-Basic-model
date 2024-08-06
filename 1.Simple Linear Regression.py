#!/usr/bin/env python
# coding: utf-8

# In[31]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


#importing the necessary dataset
data = pd.read_csv('Salary_Data.csv')
data


# In[10]:


#setting up independent variable, as yearseverperience
independent = data[["YearsExperience"]]


# In[11]:


#setting up salaru as dependent variable
dependent = data[["Salary"]]


# In[12]:


#importing training test split
from sklearn.model_selection import train_test_split


# In[14]:


#assinging data fpr training and testing split, 70 30
x_train,x_test,y_train,y_test = train_test_split(independent,dependent,test_size = .30,random_state=0)


# In[15]:


#importing linear regression from sklearn, and renaming it
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[16]:


#calculating weight of equation
weight = regressor.coef_
weight


# In[32]:


#calculating bias of equation
bias = regressor.intercept_
bias


# In[19]:


#prediction of r2score
y_pred = regressor.predict(x_test)
from sklearn.metrics import r2_score
r_score = r2_score(y_test,y_pred)
r_score


# In[20]:


#saving as a model, as a sav file
import pickle
filename = "finalized_model_linear.sav"


# In[23]:


#saves the machine learning model in binary mode
pickle.dump(regressor, open(filename,'wb'))


# In[29]:


import numpy as np
import pickle

# Load the model
loaded_model = pickle.load(open("finalized_model_linear.sav", 'rb'))

# Prepare the input data
input_data = np.array([7]).reshape(1, -1)

# Make the prediction
result = loaded_model.predict(input_data)




# In[30]:


#printing the result
result


# In[ ]:




