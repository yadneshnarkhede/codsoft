#!/usr/bin/env python
# coding: utf-8

# # Task-1 : Titanic Survival Prediction
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df =pd.read_csv("tested.csv")


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df['Survived'].value_counts()


# In[7]:


#let's Visualize the count of Survivals wrt pclass
sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[8]:


df["Sex"]


# In[9]:


#let's Visualize the count of Survivals wrt Gender
sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[10]:


#look at survival rate by sex
df.groupby('Sex')[['Survived']].mean()


# In[11]:


df['Sex'].unique()


# In[12]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex'] = labelencoder.fit_transform(df['Sex'])
df.head()


# In[13]:


df['Sex'], df['Survived']


# In[14]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[15]:


df.isna().sum()


# In[16]:


#After dropping non required column
df=df.drop(['Age'], axis=1)


# In[17]:


df_final =df 
df_final.head(10)


# In[18]:


X = df[['Pclass','Sex']]
Y = df['Survived']


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[20]:


from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)


# MODEL PREDICTION

# In[21]:


print(Y_test)


# In[22]:


import warnings 
warnings.filterwarnings("ignore")

res = log.predict([[2,0]])

if(res==0):
    print("so sorry! Not Survived")
else:
    print("Survived")

