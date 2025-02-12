#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv(r"C:\Users\MRUH\Downloads\Wholesale customers data.csv")
df.head()


# In[5]:


df


# In[6]:


df.info()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[8]:


sns.countplot(x=df['Channel'])
plt.show()


# In[9]:


sns.countplot(x=df['Region'])
plt.show()


# In[10]:


sns.countplot(x=df['Milk'])
plt.show()


# In[11]:


sns.countplot(x=df['Frozen'])
plt.show()


# In[12]:


sns.countplot(x=df['Detergents_Paper'])
plt.show()


# In[13]:


import warnings
warnings.filterwarnings('ignore')
sns.distplot(x=df['Detergents_Paper'])
plt.show()


# In[14]:


sns.distplot(x=df['Grocery'])
plt.show()


# In[15]:


sns.distplot(x=df['Fresh'])
plt.show()


# In[16]:


df.isnull().sum()


# In[17]:


df.min()


# In[18]:


df.max()


# In[19]:


df.drop(['Channel','Region'],axis=1,inplace=True)


# In[20]:


df


# In[22]:


from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler()
X=stscaler.fit_transform(df)


# In[23]:


X


# In[24]:


import scipy.cluster.hierarchy as sch


# In[27]:


plt.figure(figsize=(20,6))
dendo = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer data')
plt.ylabel('Eucl Distance')
plt.show()


# In[29]:


len(set(dendo['color_list']))-1


# In[30]:


from sklearn.cluster import AgglomerativeClustering


# In[31]:


model =  AgglomerativeClustering(n_clusters=3)
cluster=model.fit_predict(X)


# In[32]:


cluster


# In[34]:


cluster.shape


# In[35]:


df


# In[37]:


group_num=pd.DataFrame(cluster,columns=['Group'])
group_num


# In[39]:


# kmeans


# In[40]:


X


# In[41]:


from sklearn.cluster import KMeans

wcss=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[42]:


wcss


# In[43]:


plt.plot(range(2,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[48]:


kmeans = KMeans(n_clusters = 4, random_state = 99)
groups = model.fit_predict(X)
groups


# In[49]:


groups.shape


# In[53]:


groups_num=pd.DataFrame(groups,columns=['Group'])
group_num


# In[55]:


cust_kmeans_data=pd.concat([df,group_num],axis=1)
cust_kmeans_data


# In[ ]:




