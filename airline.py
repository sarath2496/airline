# -*- coding: utf-8 -*-
"""Airline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gm1c5_f2TSRWljrpPxnVbG5P60MYUN2L
"""

import pandas as pd
df=pd.read_csv('brand-mapping.csv')

df.rename(columns = {'HK':'POS'}, inplace = True)
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

df.head()

df.head()



df.info()

df.head(n=100)


bm=df.head(n=500)


bm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


train=bm.head(n=450)



train.head()

train.isnull()




sns.heatmap(train.isnull())


plt.hist(bm['FareBasisCode'])


print(bm.dtypes)



bm=df.drop(['POS', 'OriginCountry','DestinationCountry','BookingCode','Cabin','A','B','C','MilesBand','LastUpdated'], axis=1)


bm.head()

bm.loc[df['FareBasisCode']=='ASSOSSA1']



bm.drop_duplicates('FareBasisCode')

bm.head()

print(bm.shape)

bm['FareBasisCode'].replace('A',1)


expand=bm['FareBasisCode'].str.split('', expand=True)


expand=expand.replace('A','1')

expand.head()

expand=expand.replace('B',2)

expand.head()



expand=expand.replace('C',3)
expand=expand.replace('D',4)
expand=expand.replace('E',5)



expand.head()




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




expand.head()




expand.columns=['A','B','C','D','E','F','G','H','I','J']




expand.head()





expand=expand.fillna(value=0)



expand.tail()




expand['CODE']=expand['A'].astype(str)+expand['B'].astype(str)+expand['C'].astype(str)+expand['D'].astype(str)+expand['E'].astype(str)+expand['F'].astype(str)+expand['G'].astype(str)+expand['H'].astype(str)+expand['I'].astype(str)+expand['J'].astype(str)




expand.head()



expand


expand=expand.drop(['A','B','C','D','E','F','G','H','I','J'],axis=1)



expand.head()
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
clean_dataset(expand)


# In[45]:


frames=[bm,expand]
final=pd.concat(frames,axis=1,join='inner')




final.head()





final.info()




final





final['CODE']=final['CODE'].astype(int)





final.info()

final.head()

import matplotlib.pyplot as plt
x=final.CODE
y=final.BrandID
x
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

arr=X_train.values
xtrain=arr.reshape(-1,1)
xtrain

arr=y_train.values
ytrain=arr.reshape(-1,1)
ytrain

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

np.any(np.isnan(xtrain))
np.all(np.isfinite(xtrain))



clean_dataset(expand)

final.head()