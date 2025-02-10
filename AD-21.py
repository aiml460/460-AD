#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv(r"C:\Users\MRUH\Downloads\Wholesale customers data.csv")
df.head()


# In[38]:


df


# In[5]:


df.info()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


sns.countplot(x=df['Channel'])
plt.show()


# In[11]:


sns.countplot(x=df['Region'])
plt.show()


# In[14]:


sns.countplot(x=df['Milk'])
plt.show()


# In[15]:


sns.countplot(x=df['Frozen'])
plt.show()


# In[17]:


sns.countplot(x=df['Detergents_Paper'])
plt.show()


# In[26]:


import warnings
warnings.filterwarnings('ignore')
sns.distplot(x=df['Detergents_Paper'])
plt.show()


# In[27]:


sns.distplot(x=df['Grocery'])
plt.show()


# In[34]:


sns.distplot(x=df['Fresh'])
plt.show()


# In[30]:


df.isnull().sum()


# In[ ]:


import StandardScalar

