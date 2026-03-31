#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import os
import json
import sys
import joblib
sys.path.append('../')
from utils.compute_metrics import compute_rmse


# In[2]:


def svd_model(matrix,user_bias):
    #remove user bias for the model
    R_demeaned  = matrix.to_numpy() - user_bias.reshape(-1,1)
    U, sigma ,Vt = svds(R_demeaned,k=45)
    #convert sigmal from 1D array to diagonal matrix
    sigma_diagno = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma_diagno), Vt) + user_bias.reshape(-1, 1)
    return predicted_ratings


# In[3]:


def save_result(rmse,predicted_ratings):
    os.makedirs("../reports",exist_ok=True)
    np.save("../reports/svd_predictions.npy",predicted_ratings)
    print("Saved svd_predictions.npy")


# In[4]:


#load data
test_data = pd.read_csv('../processed/test_data.csv')
user_index  = joblib.load('../processed/user_index.pkl')
movie_index = joblib.load('../processed/movie_index.pkl')


# In[5]:


#Train data to a matrix
pivot_table = pd.DataFrame(np.load('../processed/svd_matrix.npy'))
#get user bias rating perference
user_bias = np.mean(pivot_table,axis=1).values
pivot_table.fillna(pivot_table.mean(),inplace=True)


# In[6]:


predicted_ratings = svd_model(pivot_table,user_bias)


# In[7]:


rmse, _, _ = compute_rmse(test_data,predicted_ratings, user_index, movie_index)


# In[8]:


print("RMSE",rmse)


# In[9]:


save_result(rmse,predicted_ratings)

