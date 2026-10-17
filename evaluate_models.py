"""
Comprehensive Model Evaluation Script
Re-evaluates both models on a common train/test split from the UCI Concrete dataset.
Diagnoses ensemble member scaling issues and extracts model details.
"""

import os
import sys
import types
import pickle
import warnings

# Suppress verbose logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Backward-compatibility shim for sklearn 1.4+
if 'sklearn.ensemble._gb_losses' not in sys.modules:
    mod = types.ModuleType('sklearn.ensemble._gb_losses')
    class LeastSquaresError:
        pass
    mod.LeastSquaresError = LeastSquaresError
    sys.modules['sklearn.ensemble._gb_losses'] = mod

try:
    from sklearn.tree import BaseDecisionTree
    if not hasattr(BaseDecisionTree, 'monotonic_cst'):
        BaseDecisionTree.monotonic_cst = None
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load the UCI Concrete Dataset
print("=" * 70)
print("STEP 1: Loading UCI Concrete Compressive Strength Dataset")
print("=" * 70)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
try:
    df = pd.read_excel(url)
except Exception:
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Concrete_Data.xls')
    if os.path.exists(local_path):
        df = pd.read_excel(local_path)
    else:
        print("ERROR: Could not download or find the UCI dataset.")
        sys.exit(1)

print(f"  Dataset shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

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

print(f"  Renamed columns: {list(df.columns)}")
print(f"  Number of records: {len(df)}")
print(f"  Target (strength) statistics:")
print(f"    Min:  {df['strength'].min():.2f} MPa")
print(f"    Max:  {df['strength'].max():.2f} MPa")
print(f"    Mean: {df['strength'].mean():.2f} MPa")
print(f"    Std:  {df['strength'].std():.2f} MPa")

# 2. Create a Common Train/Test Split
print("\n" + "=" * 70)
print("STEP 2: Creating Common Train/Test Split (80/20, random_state=42)")
print("=" * 70)

X = df[['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_agg', 'fine_agg', 'age']].values
y = df['strength'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Test set:  {X_test.shape[0]} samples")
print(f"  Test set target variance: {np.var(y_test):.2f} MPa^2")

# 3. Load Models
print("\n" + "=" * 70)
print("STEP 3: Loading Saved Models")
print("=" * 70)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')

with open(os.path.join(MODEL_DIR, 'Regression.pkl'), 'rb') as f:
    model_regression = pickle.load(f)

try:
    from sklearn._loss.loss import HalfSquaredError
    if hasattr(model_regression, 'estimators_'):
        for est in model_regression.estimators_:
            if est.__class__.__name__ == 'GradientBoostingRegressor':
                est._loss = HalfSquaredError()
except Exception:
    pass

print(f"  Regression model type: {type(model_regression).__name__}")
print(f"  Estimators: {[type(e).__name__ for e in model_regression.estimators_]}")
print(f"  Weights:    {model_regression.weights}")

with open(os.path.join(MODEL_DIR, 'NeuralNetwork.pkl'), 'rb') as f:
    model_nn = pickle.load(f)

print(f"  NN model type: {type(model_nn).__name__}")

# 4. Evaluate Ensemble on Common Test Set
print("\n" + "=" * 70)
print("STEP 4: Evaluating Ensemble Voting Regressor on Common Test Set")
print("=" * 70)

reg_preds = model_regression.predict(X_test)
reg_mae = mean_absolute_error(y_test, reg_preds)
reg_mse = mean_squared_error(y_test, reg_preds)
reg_r2 = r2_score(y_test, reg_preds)

print(f"  MAE:  {reg_mae:.4f} MPa")
print(f"  MSE:  {reg_mse:.4f} MPa^2")
print(f"  R2:   {reg_r2:.4f} ({reg_r2*100:.2f}%)")
print(f"  RMSE: {np.sqrt(reg_mse):.4f} MPa")

reg_below_0 = np.sum(reg_preds < 0)
reg_above_120 = np.sum(reg_preds > 120)
print(f"  Predictions < 0 MPa:   {reg_below_0} / {len(reg_preds)}")
print(f"  Predictions > 120 MPa: {reg_above_120} / {len(reg_preds)}")

# 5. Evaluate Each Ensemble Member Individually
print("\n" + "=" * 70)
print("STEP 5: Evaluating Each Ensemble Member Individually")
print("=" * 70)

for i, est in enumerate(model_regression.estimators_):
    name = type(est).__name__
    try:
        est_preds = est.predict(X_test)
        est_mae = mean_absolute_error(y_test, est_preds)
        est_mse = mean_squared_error(y_test, est_preds)
        est_r2 = r2_score(y_test, est_preds)
        est_min = np.min(est_preds)
        est_max = np.max(est_preds)
        below_0 = np.sum(est_preds < 0)
        above_120 = np.sum(est_preds > 120)
        weight = model_regression.weights[i] if model_regression.weights else 'equal'
        print(f"\n  [{i+1}] {name} (weight={weight})")
        print(f"      MAE: {est_mae:.4f}  MSE: {est_mse:.4f}  R2: {est_r2:.4f}")
        print(f"      Pred range: [{est_min:.2f}, {est_max:.2f}]")
        print(f"      Out-of-range: {below_0} below 0, {above_120} above 120")
    except Exception as e:
        print(f"\n  [{i+1}] {name}: ERROR - {e}")

# 6. Evaluate Neural Network on Common Test Set
print("\n" + "=" * 70)
print("STEP 6: Evaluating Neural Network on Common Test Set")
print("=" * 70)

try:
    nn_preds = model_nn.predict(X_test, verbose=0)
    nn_preds_flat = np.squeeze(nn_preds)
    nn_mae = mean_absolute_error(y_test, nn_preds_flat)
    nn_mse = mean_squared_error(y_test, nn_preds_flat)
    nn_r2 = r2_score(y_test, nn_preds_flat)
    print(f"  MAE:  {nn_mae:.4f} MPa")
    print(f"  MSE:  {nn_mse:.4f} MPa^2")
    print(f"  R2:   {nn_r2:.4f} ({nn_r2*100:.2f}%)")
    print(f"  RMSE: {np.sqrt(nn_mse):.4f} MPa")
    print(f"  Pred range: [{np.min(nn_preds_flat):.2f}, {np.max(nn_preds_flat):.2f}]")
except Exception as e:
    print(f"  ERROR: {e}")
    nn_mae = nn_mse = nn_r2 = None

# 7. Extract NN Architecture
print("\n" + "=" * 70)
print("STEP 7: Neural Network Architecture")
print("=" * 70)

try:
    model_nn.summary()
except Exception as e:
    print(f"  Could not print summary: {e}")

try:
    for i, layer in enumerate(model_nn.layers):
        config = layer.get_config()
        print(f"  Layer {i}: {layer.__class__.__name__} - {config}")
except Exception as e:
    print(f"  Could not inspect layers: {e}")

# 8. Summary
print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

print(f"Dataset: UCI Concrete Compressive Strength")
print(f"Records: {len(df)}")
print(f"Common test split: 20% ({X_test.shape[0]} samples), random_state=42")
print(f"Test set target variance: {np.var(y_test):.2f} MPa^2")
print()
print(f"  Ensemble VR:  MAE={reg_mae:.4f}  MSE={reg_mse:.4f}  R2={reg_r2:.4f}")
if nn_r2 is not None:
    print(f"  Neural Net:   MAE={nn_mae:.4f}  MSE={nn_mse:.4f}  R2={nn_r2:.4f}")
print()
print("Ensemble Members (actual from pkl):")
for i, est in enumerate(model_regression.estimators_):
    print(f"  {i+1}. {type(est).__name__} (weight={model_regression.weights[i]})")

print("\n--- END OF EVALUATION ---")
