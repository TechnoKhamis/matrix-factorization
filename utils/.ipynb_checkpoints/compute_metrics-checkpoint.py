#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def compute_rmse(test_data,predicted_ratings,user_index,movie_index):
    actual= []
    predicted = []
    for _,row in test_data.iterrows():
        rating = row["Rating"]
        userID = row["UserID"]
        movieID = row["MovieID"]
        if userID in user_index and movieID in movie_index:
            user_idx = user_index[userID]
            movie_idx = movie_index[movieID]
            pred = predicted_ratings[user_idx][movie_idx]
            pred = np.clip(pred,1,5)
            actual.append(rating)
            predicted.append(pred)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return rmse

