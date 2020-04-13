#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[22]:


diabetes = load_diabetes()


# In[23]:


data = diabetes.data


# In[24]:


data.shape


# In[25]:


type(data)


# In[28]:


data = pd.DataFrame(data=data, columns = diabetes.feature_names)
data.head()


# In[29]:


data.info()


# In[30]:


data.describe()


# In[31]:


data['target']= diabetes.target
data.head()


# In[32]:


diabetes.DESCR


# In[35]:


data.isnull().sum()


# In[36]:


sns.pairplot(data)


# In[41]:


data.corr()


# In[42]:


corrmat = data.corr()


# In[43]:


fig, ax = plt.subplots(figsize = (16,10))
sns.heatmap(corrmat, annot = True, annot_kws={'size':10})


# In[44]:


def getCorrelatedFeature(corrdata, threshold):
    feature = []
    value = []
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshold:
            feature.append(index)
            value.append(corrdata[index])
            
    df = pd.DataFrame(data=value, index= feature, columns= ['Corr value'])
    return df


# In[46]:


threshold = 0.40
corr_value = getCorrelatedFeature(corrmat['target'], threshold)
corr_value


# In[47]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[48]:


sns.pairplot(correlated_data)
plt.tight_layout()


# In[64]:


x = correlated_data.drop(labels=['target'], axis = 1)
y = correlated_data['target']
x.head()


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[51]:


x_train.shape


# In[52]:


x_test.shape


# In[53]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[54]:


y_predict = model.predict(x_test)


# In[55]:


df = pd.DataFrame(data = [y_predict, y_test])
df.T


# In[57]:


from sklearn.metrics import r2_score


# In[58]:


score = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print('r2_score: ',score)
print('mae', mae)
print('mse', mse)


# In[ ]:




