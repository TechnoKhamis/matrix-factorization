# Movie Recommender System
### SVD & PMF Matrix Factorization · MovieLens 1M · Reboot 01

A movie recommendation engine built from scratch using two matrix factorization techniques — Singular Value Decomposition (SVD) and Probabilistic Matrix Factorization (PMF) — trained on the MovieLens 1M dataset. Includes an interactive Streamlit dashboard for real-time personalized recommendations comparing both models side by side.

---

## The Problem

User-item rating matrices are extremely sparse — in this dataset, sparsity exceeds 95%. Most users have only rated a tiny fraction of available movies. Simple approaches like average ratings or content filtering fail here because they can't capture the hidden structure of user preferences.

Matrix factorization solves this by learning **latent factors** — hidden representations of both users and movies in a shared lower-dimensional space. The dot product of a user's latent vector and a movie's latent vector gives a predicted rating, even for movies the user has never seen.

---

## Results

| Model | RMSE | Target | Status |
|-------|------|--------|--------|
| SVD   | 0.8899 | ≤ 0.90 | ✅ |
| PMF   | 0.8462 | ≤ 0.85 | ✅ |
| PMF improvement over SVD | 5.0% | ≥ 5% | ✅ |

---

## Approach

### Data Preprocessing

The MovieLens 1M dataset contains 1,000,209 ratings from 6,040 users across 3,706 movies. Ratings are on a 1–5 scale.

- Data loaded with pandas from `.dat` files (double-colon separated)
- Train/test split: **99.5% train, 0.5% test** with `random_state=42`
- Separate matrices built for SVD (NaN for missing) and PMF (0 for missing)
- User and movie index mappings saved as `.pkl` files for consistent lookup
- Normalized user-item matrix saved to `processed/user_item_matrix.csv`

### SVD Model

SVD decomposes the user-item matrix R into three matrices:

```
R ≈ U × Σ × Vt
```

Where U captures user latent factors, Σ captures the singular values (importance of each factor), and Vt captures movie latent factors.

**Key design decisions:**
- User bias (mean rating per user) is removed **before** decomposition and added back after reconstruction. This was a critical fix — computing the mean after `fillna` dilutes per-user averages with column means, which inflated RMSE significantly
- `scipy.sparse.linalg.svds` with **k=45** latent factors
- Predictions clipped to [1, 5]

### PMF Model

PMF iteratively learns two factor matrices U (users) and V (movies) using stochastic gradient descent, based on the Salakhutdinov & Mnih (2007) formulation.

Unlike SVD which is a one-shot decomposition, PMF learns from each individual rating and updates the factors to reduce prediction error. It also includes **user bias** and **item bias** terms that capture global rating tendencies — a user who always rates high, or a movie that consistently gets low ratings.

**Update rules (per rating):**
```
error         = actual_rating - predicted_rating
U_i          += α * (error * V_j  - λ * U_i)
V_j          += α * (error * U_i  - λ * V_j)
user_bias[i] += α * (error        - λ * user_bias[i])
item_bias[j] += α * (error        - λ * item_bias[j])
```

**Final hyperparameters:**
```
num_factors        = 50
learning_rate (α)  = 0.005
regularization (λ) = 0.06
epochs             = 40
```

Regularization prevents overfitting by penalizing large factor values — without it, the model memorizes training ratings instead of generalizing to unseen ones.

**Why PMF beats SVD:**
SVD treats all missing values identically and has no iterative learning loop. PMF's bias terms capture that some users are consistently harsh raters and some movies are universally loved or hated, producing more precise predictions on the test set.

### Recommendation Generation

`generate_recommendations(user_id, model, top_n=10)` in `utils/recommendation.py`:

1. Looks up the user's row index from `user_index`
2. Retrieves that user's full predicted rating vector from the model's prediction matrix
3. Sorts all movies by predicted rating (descending)
4. Maps movie indices back to MovieIDs and merges with movie titles
5. Returns top-N as a DataFrame and saves to `reports/user_<id>_recommendations.csv`

---

## Project Structure

```
matrix-factorization/
│
├── data/
│   ├── ratings.dat         # 1M ratings (UserID::MovieID::Rating::Timestamp)
│   ├── users.dat           # Demographics
│   └── movies.dat          # Titles and genres
│
├── models/
│   ├── svd_model.py        # SVD training with user bias removal and scipy.svds
│   └── pmf_model.py        # PMF class with SGD, bias terms, convergence tracking
│
├── utils/
│   ├── data_loader.py      # Load raw data, train/test split, save CSVs
│   ├── matrix_creation.py  # Build pivot matrix, save user/movie index pkl files
│   ├── recommendation.py   # generate_recommendations(), comparison plots
│   └── compute_metrics.py  # RMSE computation against test set
│
├── reports/
│   ├── model_metrics.json          # SVD RMSE, PMF RMSE, improvement %
│   ├── svd_predictions.npy         # Full SVD predicted rating matrix
│   ├── pmf_predictions.npy         # Full PMF predicted rating matrix
│   ├── pmf_factors/
│   │   ├── U.npy                   # User latent factor matrix (6040 x 50)
│   │   └── V.npy                   # Movie latent factor matrix (3705 x 50)
│   ├── pmf_convergence.png         # MSE vs epoch during PMF training
│   ├── rmse_comparison.png         # SVD vs PMF RMSE bar chart
│   ├── predicted_vs_actual.png     # Scatter plots of predictions vs actuals
│   ├── user_comparison.png         # SVD vs PMF recommendations for a user
│   ├── top_recommendations.png     # Most frequently recommended movies
│   └── user_<id>_recommendations.csv
│
├── processed/
│   ├── user_item_matrix.csv    # Normalized interaction matrix
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── user_index.pkl
│   ├── movie_index.pkl
│   ├── svd_matrix.npy
│   └── pmf_matrix.npy
│
├── app.py                              # Streamlit dashboard
├── Movie_Recommender_System.ipynb      # EDA + model evaluation notebook
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/TechnoKhamis/matrix-factorization.git
cd matrix-factorization

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Download the MovieLens 1M dataset

Download from: https://grouplens.org/datasets/movielens/1m/

Place `ratings.dat`, `users.dat`, and `movies.dat` inside the `data/` folder.

---

## How to Run

Run each step in order from the project root:

```bash
# Step 1 — Load raw data, split train/test, save CSVs
python utils/data_loader.py

# Step 2 — Build user-item matrix and index mappings
python utils/matrix_creation.py

# Step 3 — Train SVD (fast, a few seconds)
python models/svd_model.py

# Step 4 — Train PMF (40 epochs, a few minutes)
python models/pmf_model.py

# Step 5 — Generate recommendations and save plots
python utils/recommendation.py

# Step 6 — Launch the Streamlit dashboard
streamlit run app.py
```

---

## Streamlit Dashboard

Input any User ID (1–6040) and hit **Generate** to see:

- The user's highest-rated movies from their watch history
- Top 10 recommendations from the SVD model
- Top 10 recommendations from the PMF model
- Side-by-side horizontal bar chart comparing predicted ratings from both models
- Global model performance metrics (RMSE, improvement %)

The dashboard handles invalid user IDs gracefully with an error message. All data loads at startup and recommendations update dynamically on each input.

---

## Key Insights from EDA

- Ratings skew positive — 4s are the most common by far, 1s the least
- The dataset is over 95% sparse — most users have never rated most movies
- A small group of power users have rated over 1,500 movies each
- Drama and Comedy dominate by volume, but Film-Noir and Documentary score highest on average
- This sparsity is exactly why latent factor models outperform simpler approaches

---

## Resources

- [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
- [Probabilistic Matrix Factorization — Salakhutdinov & Mnih (2007)](https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)
- [scipy.sparse.linalg.svds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
