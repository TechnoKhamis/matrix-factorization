#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import joblib


# In[2]:


# Load Data
try:
    movies_pd = pd.read_csv('../data/movies.dat', sep='::', encoding='latin-1',engine='python',names=["MovieID","Title","Genres"])
    users_pd = pd.read_csv('../data/users.dat', sep='::', encoding='latin-1',engine='python',names=["UserID","Gender","Age","Occupation","Zip-code"])
    ratings_pd = pd.read_csv('../data/ratings.dat', sep='::', encoding='latin-1',engine='python',names=["UserID","MovieID","Rating","Timestamp"])
except Exception as e:
    print(f"Failed to load data: {e}")


# In[3]:


#Create train and test data from rating table
train_data, test_data = train_test_split(ratings_pd, test_size=0.005, random_state=42)
#create a matrix from the train data
train_matrix = train_data.pivot_table(index="UserID", columns="MovieID", values="Rating")


# In[4]:


#scale training matrix
scaler = MinMaxScaler()
normalized_matrix = pd.DataFrame(scaler.fit_transform(train_matrix), index=train_matrix.index, columns=train_matrix.columns)


# In[5]:


normalized_matrix.to_csv("../processed/user_item_matrix.csv")


# In[6]:


test_data.to_csv('../processed/test_data.csv', index=False)


# In[7]:


train_data.to_csv('../processed/train_data.csv', index=False)

