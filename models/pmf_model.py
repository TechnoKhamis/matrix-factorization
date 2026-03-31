#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib
import sys

sys.path.append('../')
from utils.compute_metrics import compute_rmse


# In[2]:


class PMF:
    def __init__(self, num_factors=50, learning_rate=0.005, reg=0.06, epochs=40):
        self.num_factors   = num_factors
        self.learning_rate = learning_rate
        self.reg           = reg
        self.epochs        = epochs
        self.reports_dir='../reports'

    def train(self, R):
        self.num_users, self.num_items = R.shape
        np.random.seed(42)
        self.global_mean = R[R > 0].mean()
        self.U         = np.random.normal(0, 0.01, (self.num_users, self.num_factors))
        self.V         = np.random.normal(0, 0.01, (self.num_items, self.num_factors))
        self.user_bias = np.zeros(self.num_users)
        self.item_bias = np.zeros(self.num_items)
        self.losses    = []

        xs      = R.nonzero()[0].astype(np.int32)
        ys      = R.nonzero()[1].astype(np.int32)
        ratings = R[xs, ys].astype(np.float64)

        for epoch in range(self.epochs):
            idx = np.random.permutation(len(xs))
            for k in idx:
                i = xs[k]
                j = ys[k]
                pred  = self.global_mean + self.user_bias[i] + self.item_bias[j] + np.dot(self.U[i], self.V[j])
                error = ratings[k] - pred
                U_old = self.U[i].copy()
                self.U[i]         += self.learning_rate * (error * self.V[j]  - self.reg * self.U[i])
                self.V[j]         += self.learning_rate * (error * U_old      - self.reg * self.V[j])
                self.user_bias[i] += self.learning_rate * (error              - self.reg * self.user_bias[i])
                self.item_bias[j] += self.learning_rate * (error              - self.reg * self.item_bias[j])

            preds = np.clip(
                self.global_mean + self.user_bias[xs] + self.item_bias[ys] +
                (self.U[xs] * self.V[ys]).sum(axis=1), 1, 5
            )
            mse = mean_squared_error(ratings, preds)
            self.losses.append(mse)
            print(f"Epoch {epoch+1}/{self.epochs}, MSE: {mse:.4f}")

    def predict(self):
        return np.clip(
            self.global_mean +
            self.user_bias.reshape(-1, 1) +
            self.item_bias.reshape(1, -1) +
            np.dot(self.U, self.V.T), 1, 5
        )
    def save_convergence_plot(self):
        os.makedirs(self.reports_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.losses) + 1), self.losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('PMF Convergence — MSE vs Epoch')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.reports_dir}/pmf_convergence.png')
        plt.close()
        print("Saved pmf_convergence.png")
        
    def save_factors(self):
        os.makedirs(f'{self.reports_dir}/pmf_factors', exist_ok=True)
        np.save(f'{self.reports_dir}/pmf_factors/U.npy', self.U)
        np.save(f'{self.reports_dir}/pmf_factors/V.npy', self.V)
        print(f"U shape: {self.U.shape}")
        print(f"V shape: {self.V.shape}")

    def save_all(self, predicted):
        np.save(f'{self.reports_dir}/pmf_predictions.npy', predicted)
        print("Saved pmf_predictions.npy")
        self.save_convergence_plot()
        self.save_factors()
        


# In[3]:


# Load data
test_data = pd.read_csv('../processed/test_data.csv')

user_index  = joblib.load('../processed/user_index.pkl')
movie_index = joblib.load('../processed/movie_index.pkl')

R = np.load('../processed/pmf_matrix.npy')


# In[4]:


# Train
pmf = PMF(num_factors=50, learning_rate=0.005, reg=0.06, epochs=40)
pmf.train(R)

# Evaluate
predicted = pmf.predict()
predicted = np.clip(predicted, 1, 5)

rmse, _, _ = compute_rmse(test_data,predicted,user_index, movie_index)

print("RMSE",rmse)


# In[5]:


#save predictions convergence and factors
pmf.save_all(predicted)

