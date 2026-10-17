"""
Retrain both models on the UCI Concrete dataset with a common 80/20 split.
Saves new .pkl files with proper evaluation on a shared test set.
"""

import os
import sys
import pickle
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    VotingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBRegressor

# ─── 1. Load Dataset ─────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading UCI Concrete Dataset (1030 records)")
print("=" * 70)

local_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Concrete_Data.csv')
local_xls = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Concrete_Data.xls')
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-With-R-Datasets/master/concrete.csv'

if os.path.exists(local_csv):
    df = pd.read_csv(local_csv)
elif os.path.exists(local_xls):
    df = pd.read_excel(local_xls)
else:
    df = pd.read_csv(url)
    df.to_csv(local_csv, index=False)

col_map = {
    df.columns[0]: 'cement',
    df.columns[1]: 'slag',
    df.columns[2]: 'fly_ash',
    df.columns[3]: 'water',
    df.columns[4]: 'superplasticizer',
    df.columns[5]: 'coarse_agg',
    df.columns[6]: 'fine_agg',
    df.columns[7]: 'age',
    df.columns[8]: 'strength'
}
df.rename(columns=col_map, inplace=True)

feature_cols = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_agg', 'fine_agg', 'age']
X = df[feature_cols].values
y = df['strength'].values

print(f"  Records: {len(df)}")
print(f"  Features: {feature_cols}")
print(f"  Target range: [{y.min():.2f}, {y.max():.2f}] MPa")

# ─── 2. Common Split ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Train/Test Split (80/20, random_state=42)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ─── 3. Train Individual Models ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Training Individual Models")
print("=" * 70)

models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {'model': model, 'mae': mae, 'mse': mse, 'r2': r2, 'preds': preds}
    print(f"  {name:20s}  MAE={mae:.4f}  MSE={mse:.4f}  R2={r2:.4f}")

# ─── 4. Build Voting Ensemble (tree-based only, no linear models on raw data)
print("\n" + "=" * 70)
print("STEP 4: Building Voting Ensemble (tree-based members only)")
print("=" * 70)

ensemble = VotingRegressor(
    estimators=[
        ('rf', models['RandomForest']),
        ('gb', models['GradientBoosting']),
        ('xgb', models['XGBoost']),
        ('ada', models['AdaBoost']),
        ('dt', models['DecisionTree']),
    ],
    weights=[3, 4, 9, 2, 2]
)
# The individual estimators are already fitted; fit the ensemble wrapper
ensemble.fit(X_train, y_train)

ens_preds = ensemble.predict(X_test)
ens_mae = mean_absolute_error(y_test, ens_preds)
ens_mse = mean_squared_error(y_test, ens_preds)
ens_r2 = r2_score(y_test, ens_preds)

print(f"  Ensemble VR:  MAE={ens_mae:.4f}  MSE={ens_mse:.4f}  R2={ens_r2:.4f}")
print(f"  Pred range: [{ens_preds.min():.2f}, {ens_preds.max():.2f}]")
print(f"  Out-of-range (<0 or >120): {np.sum((ens_preds < 0) | (ens_preds > 120))}")

# Cross-validation on full data
cv_scores = cross_val_score(
    VotingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)),
            ('ada', AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
            ('dt', DecisionTreeRegressor(max_depth=10, random_state=42)),
        ],
        weights=[3, 4, 9, 2, 2]
    ),
    X, y, cv=5, scoring='r2', n_jobs=-1
)
print(f"  5-Fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ─── 5. Train Neural Network ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Training Neural Network (with StandardScaler)")
print("=" * 70)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse', metrics=['mae'])

    history = nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )

    nn_preds = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_mae = mean_absolute_error(y_test, nn_preds)
    nn_mse = mean_squared_error(y_test, nn_preds)
    nn_r2 = r2_score(y_test, nn_preds)

    print(f"  Neural Net:  MAE={nn_mae:.4f}  MSE={nn_mse:.4f}  R2={nn_r2:.4f}")
    print(f"  Pred range: [{nn_preds.min():.2f}, {nn_preds.max():.2f}]")
    print(f"  Architecture: 8 -> 128(relu) -> 64(relu) -> 32(relu) -> 1(linear)")
    print(f"  Training epochs (early stop): {len(history.history['loss'])}")

    nn_trained = True
except Exception as e:
    print(f"  ERROR training NN: {e}")
    nn_trained = False

# ─── 6. Save Models ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Saving Retrained Models")
print("=" * 70)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')

# Backup old models
import shutil
for fname in ['Regression.pkl', 'NeuralNetwork.pkl']:
    src = os.path.join(MODEL_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, src + '.bak')
        print(f"  Backed up {fname} -> {fname}.bak")

# Save new ensemble
with open(os.path.join(MODEL_DIR, 'Regression.pkl'), 'wb') as f:
    pickle.dump(ensemble, f)
print(f"  Saved new Regression.pkl")

# Save NN + scaler together
if nn_trained:
    nn_bundle = {'model': nn_model, 'scaler': scaler}
    with open(os.path.join(MODEL_DIR, 'NeuralNetwork.pkl'), 'wb') as f:
        pickle.dump(nn_bundle, f)
    print(f"  Saved new NeuralNetwork.pkl (with scaler)")

# Also update trained_model.pkl to match Regression.pkl
shutil.copy2(
    os.path.join(MODEL_DIR, 'Regression.pkl'),
    os.path.join(MODEL_DIR, 'trained_model.pkl')
)
print(f"  Updated trained_model.pkl")

# ─── 7. Final Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL METRICS (Common Test Set: 206 samples, random_state=42)")
print("=" * 70)

print(f"\n  Dataset: UCI Concrete Compressive Strength (1030 records)")
print(f"  Split: 80/20, random_state=42")
print(f"  Test set variance: {np.var(y_test):.2f} MPa^2\n")
print(f"  {'Model':<30s} {'MAE':>8s} {'MSE':>10s} {'R2':>8s}")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
print(f"  {'Ensemble Voting Regressor':<30s} {ens_mae:>8.4f} {ens_mse:>10.4f} {ens_r2:>8.4f}")
if nn_trained:
    print(f"  {'Neural Network (scaled)':<30s} {nn_mae:>8.4f} {nn_mse:>10.4f} {nn_r2:>8.4f}")
print(f"\n  Ensemble 5-Fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n  Ensemble Members:")
for name, w in zip(['RandomForest', 'GradientBoosting', 'XGBoost', 'AdaBoost', 'DecisionTree'], [3, 4, 9, 2, 2]):
    r = results[name]
    print(f"    {name:<20s} weight={w}  MAE={r['mae']:.4f}  R2={r['r2']:.4f}")

print("\n--- RETRAINING COMPLETE ---")
