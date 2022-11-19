#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df= pd.read_csv('brand-mapping.csv')


# In[2]:


df.head()


# In[3]:


df.rename(columns = {'HK':'POS'}, inplace = True)


# In[4]:


df.head()


# In[5]:


df.rename(columns = {'AE':'OriginCountry'}, inplace = True)
df.rename(columns = {'AF':'DestinationCountry'}, inplace = True)
df.rename(columns = {'EK':'Carrier'}, inplace = True)
df.rename(columns = {'A':'BookingCode'}, inplace = True)
df.rename(columns = {'ASSOSKW1':'FareBasisCode'}, inplace = True)
df.rename(columns = {'First':'Cabin'}, inplace = True)
df.rename(columns = {'FZ':'A'}, inplace = True)
df.rename(columns = {'KW':'B'}, inplace = True)
df.rename(columns = {'AF.1':'C'}, inplace = True)
df.rename(columns = {'1001-2000':'MilesBand'}, inplace = True)
df.rename(columns = {'1118772':'BrandID'}, inplace = True)
df.rename(columns = {'Unnamed: 12':'LastUpdated'}, inplace = True)


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.head(n=100)


# In[9]:


bm=df.head(n=500)


# In[10]:


bm


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


print(bm.dtypes)


# In[13]:


bm=df.drop(['POS', 'OriginCountry','DestinationCountry','BookingCode','A','B','C','LastUpdated'], axis=1)


# In[14]:


bm.head()


# In[15]:


bm.drop_duplicates('FareBasisCode')


# In[16]:


bm.head()


# In[17]:


print(bm.shape)


# In[18]:


expand=bm['FareBasisCode'].str.split('', expand=True)


# In[19]:


expand.head()


# In[20]:


frames=[bm,expand]
final=pd.concat(frames,axis=1,join='inner')


# In[21]:


f=final.drop(['FareBasisCode'],axis=1)


# In[22]:


f.rename(columns = {0:'A'}, inplace = True)
f.rename(columns = {1:'B'}, inplace = True)
f.rename(columns = {2:'C'}, inplace = True)
f.rename(columns = {3:'D'}, inplace = True)
f.rename(columns = {4:'E'}, inplace = True)
f.rename(columns = {5:'F'}, inplace = True)
f.rename(columns = {6:'G'}, inplace = True)
f.rename(columns = {7:'H'}, inplace = True)
f.rename(columns = {8:'I'}, inplace = True)
f.rename(columns = {9:'J'}, inplace = True)


# In[23]:


f.head()


# In[24]:


final.info()


# In[25]:


f


# In[26]:


bm=f.head(1000)


# In[27]:


bm.info()


# In[28]:


bm


# In[30]:


S=bm.drop(['A','J'],axis=1)


# In[31]:


S


# In[32]:


S=pd.get_dummies(bm,columns=['Carrier','MilesBand','Cabin','B','C','D','E','F','G','H','I'])


# In[41]:


x= S.drop(['BrandID'],axis=1)
y= S['BrandID']


# In[42]:


X=x.drop(['A','J'],axis=1)


# In[43]:


X


# In[44]:


y


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[47]:


from sklearn.ensemble import RandomForestClassifier


# In[48]:


clf = RandomForestClassifier(n_estimators = 200) 


# In[49]:


clf.fit(X_train, y_train)


# In[50]:


y_pred = clf.predict(X_test)


# In[51]:


from sklearn import metrics 
print()
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# In[52]:


clf.predict([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0
]])


# In[ ]:




