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


train=bm.head(n=450)


# In[13]:


train.head()


# In[14]:


train.isnull()


# In[15]:


sns.heatmap(train.isnull())


# In[16]:


plt.hist(bm['FareBasisCode'])


# In[17]:


print(bm.dtypes)


# In[18]:


bm=df.drop(['POS', 'OriginCountry','DestinationCountry','BookingCode','Cabin','A','B','C','MilesBand','LastUpdated'], axis=1)


# In[19]:


bm.head()


# In[20]:


bm.loc[df['FareBasisCode']=='ASSOSSA1']


# In[21]:


bm.drop_duplicates('FareBasisCode')


# In[22]:


bm.head()


# In[23]:


print(bm.shape)


# In[24]:


bm['FareBasisCode'].replace('A',1)


# In[27]:


expand=bm['FareBasisCode'].str.split('', expand=True)


# In[28]:


expand=expand.replace('A','1')


# In[29]:


expand.head()


# In[30]:


expand=expand.replace('B',2)


# In[31]:


expand.head()


# In[32]:


expand=expand.replace('C',3)
expand=expand.replace('D',4)
expand=expand.replace('E',5)


# In[33]:


expand.head()


# In[34]:


expand=expand.replace('F',6)
expand=expand.replace('G',7)
expand=expand.replace('H',8)
expand=expand.replace('I',9)
expand=expand.replace('J',10)
expand=expand.replace('K',11)
expand=expand.replace('L',12)
expand=expand.replace('M',13)
expand=expand.replace('N',14)
expand=expand.replace('O',15)
expand=expand.replace('P',16)
expand=expand.replace('Q',17)
expand=expand.replace('R',18)
expand=expand.replace('S',19)
expand=expand.replace('T',20)
expand=expand.replace('U',21)
expand=expand.replace('V',22)
expand=expand.replace('W',23)
expand=expand.replace('X',24)
expand=expand.replace('Y',25)

expand=expand.replace('Z',26)


# In[35]:


expand.head()


# In[36]:


expand.columns=['A','B','C','D','E','F','G','H','I','J']


# In[37]:


expand.head()


# In[38]:


expand=expand.fillna(value=0)


# In[39]:


expand.tail()


# In[40]:


expand['CODE']=expand['A'].astype(str)+expand['B'].astype(str)+expand['C'].astype(str)+expand['D'].astype(str)+expand['E'].astype(str)+expand['F'].astype(str)+expand['G'].astype(str)+expand['H'].astype(str)+expand['I'].astype(str)+expand['J'].astype(str)


# In[41]:


expand.head()


# In[42]:


expand


# In[43]:


expand=expand.drop(['A','B','C','D','E','F','G','H','I','J'],axis=1)


# In[44]:


expand.head()


# In[45]:


frames=[bm,expand]
final=pd.concat(frames,axis=1,join='inner')


# In[46]:


final.head()


# In[47]:


final.info()


# In[48]:


final


# In[49]:


final['CODE']=final['CODE'].astype(int)


# In[50]:


final.info()


# In[51]:


x=final.CODE


# In[52]:


y=final.BrandID


# In[53]:


y


# In[54]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[55]:


X_train


# In[56]:


X_test


# In[57]:


Y_train


# In[58]:


Y_test


# In[59]:


arr=X_train.values
xtrain=arr.reshape(-1,1)


# In[60]:


arr=Y_train.values
ytrain=arr.reshape(-1,1)


# In[61]:


xtrain


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)


# In[ ]:


from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
							stop = X_set[:, 0].max() + 1, step = 0.01),
					np.arange(start = X_set[:, 1].min() - 1,
							stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
			np.array([X1.ravel(), X2.ravel()]).T).reshape(
			X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
	
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

