#!/usr/bin/env python
# coding: utf-8

# # Brain Tumour Classification

# In[48]:


'''
importing pandas and numpy for dataset and mathemetical operations
importing seaborn and matplotlib for data visualisation

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


#getting the directory

import os
os.getcwd()


# In[51]:


#calling the dataset

df=pd.read_csv('C:/Users/vansh/Desktop/pie infocomm/bt_dataset_t3.csv')


# In[52]:


df.head()


# In[53]:


df.describe()


# In[54]:


df.info()


# In[55]:


#filling all the null values with the mean

df=df.fillna(df.mean())


# In[56]:


df.info()


# In[57]:


df.columns


# In[58]:


df.dtypes


# In[83]:


df=df.replace([np.inf, -np.inf], np.nan)
df.info()


# In[75]:


#1=tumour   &   0=not tumour
sns.set(style='darkgrid')
plt.figure(figsize=(8,4))

sns.countplot(df['Target'],palette='coolwarm')


# In[76]:


df['Target'].value_counts()


# In[77]:


sns.pairplot(data=df.drop('Image',axis=1),hue='Target')


# In[78]:


df1=df.corr()
plt.figure(figsize=(12,10))

sns.heatmap(df1,)


# In[95]:


#Machine Learning part starts  
from sklearn.model_selection import train_test_split


# In[96]:



X=df.drop(['Image','Target','PSNR'],axis=1)
y=df['Target']

X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)


# In[97]:


#binary classification algorithm
from sklearn.linear_model import LogisticRegression


# In[98]:


logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[89]:


predictions=logmodel.predict(X_test)


# In[92]:


from sklearn.metrics import classification_report, confusion_matrix


# In[91]:


print(classification_report(y_test,predictions))


# In[93]:


print(confusion_matrix(y_test,predictions))


# In[100]:


print('The model is 88% accurate')


# In[ ]:




