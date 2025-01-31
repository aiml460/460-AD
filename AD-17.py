#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pandas import read_csv
import seaborn as  sns
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression


# In[23]:


dataframe=sns.load_dataset('tips')
dataframe


# In[24]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
dataframe['smoker'] = lb.fit_transform(dataframe['smoker'])
dataframe['sex'] = lb.fit_transform(dataframe['sex'])
dataframe['time'] = lb.fit_transform(dataframe['time'])
dataframe['day'] = lb.fit_transform(dataframe['day'])                                      


# In[25]:


x = dataframe.drop('tip',axis=1)
y = dataframe.tip


# In[26]:


x


# In[27]:


y


# In[28]:


test=SelectKBest(score_func=f_regression,k=3).fit(x,y)
test


# In[29]:


np.round(test.pvalues_,3)


# In[31]:


sel=SelectKBest(score_func=f_regression,k=2).fit(x,y)
sel


# In[32]:


sel.pvalues_


# In[34]:


sel.scores_


# In[36]:


dataframe.columns[np.where(sel.get_support(indices=True))]


# In[39]:


df.info()


# In[47]:


df.isnull()


# In[ ]:


#for regression: f_regression, mutual_info_regression
#From classification:chi2


# In[48]:


dataframe=sns.load_dataset('tips')
dataframe.head()


# In[ ]:




