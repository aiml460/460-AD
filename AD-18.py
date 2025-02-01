#!/usr/bin/env python
# coding: utf-8

# In[26]:


import warnings
warnings.filterwarnings('ignore')


# In[25]:


import pandas as pd
data = pd.read_csv(r"C:\Users\MRUH\Downloads\NewspaperData - NewspaperData.csv")
data.head()


# In[3]:


data.info()


# In[8]:


data.head()


# In[9]:


data.tail()


# In[19]:


data.sample(9)


# correlation

# In[22]:


data.drop('Newspaper',axis=1).corr()


# In[24]:


import seaborn as sns
sns.displot(data['daily'])


# In[ ]:




