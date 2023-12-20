#!/usr/bin/env python
# coding: utf-8

# # TASK-4: Sales Prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Sales prediction dataset.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


sns.pairplot(df, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


# In[7]:


df['TV'].plot.hist(bins=10)


# In[10]:


df['Radio'].plot.hist(bins=10, color="red", xlabel="Radio")


# In[11]:


df['Newspaper'].plot.hist(bins=10, color="orange", xlabel="Radio")


# Histogram Observation.
# 
# 
# The majority sales is the result of low advertising cost in newspaper

# In[12]:


sns.heatmap(df.corr(),annot= True)
plt.show()


# **Sales is highly coorelated with the TV**

# Lets train our model using linear regression as it is coorelated with only one variable TV

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)


# In[14]:


print(X_train)


# In[15]:


print(y_train)


# In[17]:


print(X_test)


# In[18]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[19]:


res=model.predict(X_test)
print(res)


# In[21]:


model.coef_


# In[22]:


model.intercept_


# In[24]:


0.05473199*69.2 + 7.14382225


# In[25]:


plt.plot(res)


# In[26]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 7.14382225 + 0.05473199 * X_test, 'r')
plt.show()


# ***concluding with saying that above mention solution is successfully able to predict the sales using advertising platform datasets.***
