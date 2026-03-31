#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
import os


# In[2]:


def create_rating_matrix(data,result_dir="../processed"):
    os.makedirs(result_dir,exist_ok=True)
    
    matrix = pd.pivot(data,values="Rating",index="UserID",columns="MovieID")
    
    user_index  = {uid: idx for idx, uid in enumerate(matrix.index)}
    movie_index = {mid: idx for idx, mid in enumerate(matrix.columns)}
    
    joblib.dump(user_index,  f'{result_dir}/user_index.pkl')
    joblib.dump(movie_index, f'{result_dir}/movie_index.pkl')

    pmf_matrix = matrix.fillna(0)

    np.save(result_dir+"/svd_matrix",matrix.to_numpy())
    np.save(result_dir+"/pmf_matrix",pmf_matrix.to_numpy())    


# In[3]:


train_data = pd.read_csv('../processed/train_data.csv')
create_rating_matrix(train_data)

