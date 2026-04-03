#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


# In[2]:


BASE = os.path.dirname(os.path.abspath(__file__))

svd_data    = np.load(os.path.join(BASE, '../reports/svd_predictions.npy'))
pmf_data    = np.load(os.path.join(BASE, '../reports/pmf_predictions.npy'))
user_index  = joblib.load(os.path.join(BASE, '../processed/user_index.pkl'))
movie_index = joblib.load(os.path.join(BASE, '../processed/movie_index.pkl'))
movies_pd   = pd.read_csv(os.path.join(BASE, '../data/movies.dat'), sep='::', encoding='latin-1', engine='python', names=["MovieID","Title","Genres"])
test_data   = pd.read_csv(os.path.join(BASE, '../processed/test_data.csv'))
train_data  = pd.read_csv(os.path.join(BASE, '../processed/train_data.csv'))


# In[3]:


def generate_recommendations(user_id, model, top_n=10):
    
    movies_indexs = {idx: movie for movie, idx in movie_index.items()} 
    
    if user_id not in user_index:
        return None
    
    user_idx = user_index[user_id]

    match model:
        case "svd":
            user_preds=svd_data[user_idx]
        case "pmf":
            user_preds=pmf_data[user_idx]
        case _:
            print(f"Unknown model: {model}")
            return None
        
    top_movies_indexs = np.argsort(user_preds)[::-1][:top_n]
    
    top_movies_ids = [ movies_indexs[i] for i in top_movies_indexs]
    
    top_movies = movies_pd[movies_pd["MovieID"].isin(top_movies_ids)]

    top_movies["Rating"] =  [round(user_preds[i], 2) for i in top_movies_indexs]

    os.makedirs("../reports",exist_ok=True)
    
    top_movies.to_csv(f'../reports/user_{user_id}_recommendations.csv',index=False)

    return top_movies
    


# In[4]:


def plot_user_comparison(user_id, reports_dir='../reports'):
    svd_recs = generate_recommendations(user_id, model='svd')
    pmf_recs = generate_recommendations(user_id, model='pmf')
    
    if svd_recs is None or pmf_recs is None:
        print(f"User {user_id} not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].barh(svd_recs['Title'], svd_recs['Rating'], color='blue')
    axes[0].set_title(f'SVD Recommendations — User {user_id}')
    axes[0].set_xlabel('Rating')
    
    axes[1].barh(pmf_recs['Title'], pmf_recs['Rating'], color='green')
    axes[1].set_title(f'PMF Recommendations — User {user_id}')
    axes[1].set_xlabel('Rating')
    
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/user_comparison.png')
    plt.close()
    print("Saved user_comparison.png")


# In[5]:


def plot_top_recommendations(user_ids=[1,2,3,4,5], reports_dir='../reports'):
    all_recs = []
    for uid in user_ids:
        recs = generate_recommendations(uid, model='pmf')
        if recs is not None:
            all_recs.append(recs)
    
    combined   = pd.concat(all_recs)
    top_movies = combined['Title'].value_counts().head(10)
    
    plt.figure(figsize=(12, 5))
    top_movies.plot(kind='bar', color='green')
    plt.title('Top Recommended Movies')
    plt.xlabel('Movie')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/top_recommendations.png')
    plt.close()
    print("Saved top_recommendations.png")


# In[6]:


#plot_user_comparison(user_id=1)
#plot_top_recommendations(user_ids=[1, 2, 3, 4, 5])


# In[7]:


# test data audit
test_user_id = test_data["UserID"].iloc[0]

print(test_user_id)

plot_user_comparison(user_id=test_user_id)
plot_top_recommendations(user_ids=[test_user_id])


# In[8]:


#Train data audit
train_data_id = int(train_data["UserID"].iloc[1])
plot_user_comparison(user_id=train_data_id)
plot_top_recommendations(user_ids=[train_data_id])


# In[9]:


'''

SVD mixes good recommendations with obscure films
because it doesn't handle item bias as well as PMF.

PMF's user_bias and item_bias terms allow it to
capture that User 5412 is a discerning rater who
prefers critically acclaimed films.

'''


# In[ ]:




