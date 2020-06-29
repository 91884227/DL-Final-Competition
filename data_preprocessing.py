#!/usr/bin/env python
# coding: utf-8

# # import data

# In[17]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json


# # read csv

# In[18]:


temp_ = "./raw_data/%s"


# In[19]:


df = pd.read_csv(temp_ % "train_data.csv")


# # split data

# In[4]:


y = df.label.to_numpy()


# In[5]:


X = df.title.to_numpy()


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.1, 
                                                    random_state = 42)


# # save data

# In[8]:


X_train = list(X_train)
X_test = list(X_test)
y_train = list(y_train)
y_test = list(y_test)


# In[9]:


y_train = [int(i) for i in y_train]
y_test = [int(i) for i in y_test]


# In[10]:


temp_ = "./data/%s"


# In[11]:


with open(temp_ % "X_train.json", "w") as outfile:
    json.dump(X_train, outfile)


# In[12]:


with open(temp_ % "X_test.json", "w") as outfile:
    json.dump(X_test, outfile)


# In[13]:


with open(temp_ % "y_train.json", "w") as outfile:
    json.dump(y_train, outfile)


# In[14]:


with open(temp_ % "y_test.json", "w") as outfile:
    json.dump(y_test, outfile)


# # upload data

# In[20]:


df = pd.read_csv(temp_ % "test_data.csv")


# In[22]:


X = df.title.to_numpy()


# In[26]:


X_upload = list(X)


# In[27]:


temp_ = "./data/%s"
with open(temp_ % "X_upload.json", "w") as outfile:
    json.dump(X_upload, outfile)


# In[ ]:




