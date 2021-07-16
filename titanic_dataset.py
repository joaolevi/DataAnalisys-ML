#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


path = "dataset/titanic/"


# In[3]:


dt_train = pd.read_csv(path+"train.csv")


# In[4]:


dt_train.head()


# In[5]:


dt_train.info()


# In[7]:


dt_train['Cabin'].value_counts()


# In[56]:


dt_train_wout_cabin = dt_train.drop(columns=['Cabin'])


# In[57]:


dt_train_wout_cabin.info()


# In[58]:


dt_train_wout_cabin['Age'].fillna(round(dt_train['Age'].mean(), 2), inplace=True)


# In[59]:


dt_train_wout_cabin.info()


# In[49]:


dt_train_wout_cabin.head()


# In[50]:


dt_train_wout_cabin['Ticket'].value_counts()


# In[30]:


dt_train_wout_cabin['Ticket'].value_counts().max()


# In[60]:


dt_cat = dt_train_wout_cabin['Embarked']


# In[52]:


dt_cat.head(10)


# In[61]:


#É possível ver que a coluna Embarked é uma coluna categórica
#Então, usando o método factorize() do pandas, podemos criar um array
#que atribuirá valores numéricos inteiros as categorias
dt_cat_encoded, dt_cat_categories = dt_cat.factorize()


# In[62]:


dt_cat_encoded[:10]


# In[63]:


#Nota-se que existem 3 categorias ['S', 'C', 'Q']
dt_cat_categories


# In[64]:


#Confirma-se a tese de que existem valores ausentes na coluna 'Embarked'
dt_train_wout_cabin['Embarked'].unique()


# In[65]:


#Porém, ao contar os valores para cada categoria, não aparece nenhum valor ausente
dt_train_wout_cabin['Embarked'].value_counts()


# In[66]:


#Como a soma é 889 então existem valores ausentes
dt_train_wout_cabin['Embarked'].value_counts().sum()


# In[69]:


dt_wout_na = dt_train_wout_cabin.dropna()


# In[70]:


dt_wout_na.info()


# In[71]:


dt_cat_encoded, dt_cat_categories = dt_cat.factorize()


# In[72]:


dt_cat_encoded[:10]


# In[73]:


dt_train_wout_cabin['Embarked'].unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




