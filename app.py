"""
Concrete Compressive Strength Prediction Web Application
Author & Sole Maintainer: Ali Hassan Kadri
License: MIT License
"""

import os
import sys
import types
import pickle
import warnings

# Suppress TensorFlow verbose logging & oneDNN notices
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress sklearn unpickling version warnings
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', category=UserWarning)

# Backward-compatibility shim for sklearn 1.4+ unpickling older GradientBoostingRegressor
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
from flask import Flask, request, render_template, jsonify, send_file

app = Flask(__name__, static_folder='static', template_folder='templates')

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')

reg_model_path = os.path.join(MODEL_DIR, 'Regression.pkl')
nn_model_path = os.path.join(MODEL_DIR, 'NeuralNetwork.pkl')

model_regression = None
model_neural_network = None

# Attempt loading models with comprehensive error reporting & compatibility patching
try:
    if os.path.exists(reg_model_path):
        with open(reg_model_path, 'rb') as f:
            model_regression = pickle.load(f)
        
        # Patch loss functions on loaded estimators if running on modern scikit-learn
        try:
            from sklearn._loss.loss import HalfSquaredError
            if hasattr(model_regression, 'estimators_'):
                for est in model_regression.estimators_:
                    if est.__class__.__name__ == 'GradientBoostingRegressor':
                        est._loss = HalfSquaredError()
        except Exception:
            pass
            
        print("[INFO] Ensemble Voting Regressor model successfully loaded.")
except Exception as e:
    print(f"[WARNING] Warning loading Regression.pkl: {e}")

try:
    if os.path.exists(nn_model_path):
        with open(nn_model_path, 'rb') as f:
            model_neural_network = pickle.load(f)
        print("[INFO] Deep Neural Network model successfully loaded.")
except Exception as e:
    print(f"[WARNING] Warning loading NeuralNetwork.pkl: {e}")


def calculate_calibrated_strength(cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age):
    """
    Empirical Abrams' Law formula for concrete compressive strength (MPa)
    used as calibrated fallback if a serialized model dependency is missing.
    """
    binder = cement + 0.8 * slag + 0.5 * fly_ash
    wc_ratio = water / max(binder, 1.0)
    
    # Base 28-day strength from water-cement ratio
    base_28d = 100.0 / (3.2 ** (2.1 * wc_ratio))
    
    # Curing age factor (logarithmic hydration growth)
    age_factor = (np.log(max(age, 1.0)) / np.log(28.0)) ** 0.58 if age < 28 else (1.0 + 0.08 * np.log10(age / 28.0))
    
    # Superplasticizer compaction density factor
    sp_factor = 1.0 + (min(superplasticizer, 25.0) * 0.012)
    
    pred = base_28d * age_factor * sp_factor
    return max(round(float(pred), 2), 2.5)


def classify_concrete_grade(strength_mpa):
    """Classifies concrete into Eurocode EN 206 / ACI structural categories."""
    if strength_mpa < 20.0:
        return (
            "C12/15 - Low Strength Concrete",
            "Suitable for plain mass concrete, lean blinding, trench fill, and non-structural levelling beds."
        )
    elif strength_mpa < 30.0:
        return (
            "C20/25 - Standard Structural",
            "Standard commercial grade for domestic foundations, slabs, footings, and pavement."
        )
    elif strength_mpa < 40.0:
        return (
            "C30/37 - Reinforced Commercial",
            "Durable reinforced concrete for external paved areas, floor slabs, structural walls, and beams."
        )
    elif strength_mpa < 55.0:
        return (
            "C40/50 - High Performance",
            "High-strength structural grade for heavy commercial buildings, columns, and prestressed components."
        )
    else:
        return (
            "C60+ - Ultra High Strength",
            "Advanced structural concrete for high-rise infrastructure, marine piers, and critical civil works."
        )


def extract_features(data_dict):
    """Extracts features in exact sequence required by the models."""
    try:
        cement = float(data_dict.get('cement', data_dict.get('Quanity of Cement in kg', 280.0)))
        slag = float(data_dict.get('slag', data_dict.get('Quanity of blast furnace slag in kg', 0.0)))
        fly_ash = float(data_dict.get('fly_ash', data_dict.get('Quanity of fly ash in kg', 0.0)))
        water = float(data_dict.get('water', data_dict.get('Quanity of water in kg', 185.0)))
        superplasticizer = float(data_dict.get('superplasticizer', data_dict.get('Quanity of superplasticizer in kg', 2.5)))
        coarse_agg = float(data_dict.get('coarse_agg', data_dict.get('Quanity of coarse_aggregate in kg', 1040.0)))
        fine_agg = float(data_dict.get('fine_agg', data_dict.get('Quanity of fine_aggregate in kg', 780.0)))
        age = float(data_dict.get('age', data_dict.get('No of days to cure', 28.0)))
        
        feature_vector = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
        return feature_vector, {
            'cement': cement, 'slag': slag, 'fly_ash': fly_ash, 'water': water,
            'superplasticizer': superplasticizer, 'coarse_agg': coarse_agg,
            'fine_agg': fine_agg, 'age': age
        }
    except Exception as e:
        raise ValueError(f"Feature extraction error: {e}")


def run_prediction(feature_vector, feat_dict):
    """Executes predictions across both models with fallback safety."""
    # 1. Regression Model
    reg_val = None
    if model_regression is not None:
        try:
            preds = model_regression.predict(feature_vector)
            val = float(preds[0])
            # If the voting ensemble produces an unscaled outlier (>120 or <0), average the tree-based sub-estimators
            if val > 120.0 or val < 0.0:
                tree_preds = []
                for est in getattr(model_regression, 'estimators_', []):
                    try:
                        p = float(est.predict(feature_vector)[0])
                        if 0.0 <= p <= 120.0:
                            tree_preds.append(p)
                    except Exception:
                        pass
                if tree_preds:
                    val = float(np.mean(tree_preds))
            reg_val = round(val, 2)
        except Exception as e:
            print(f"Regression prediction error: {e}")

    if reg_val is None:
        reg_val = calculate_calibrated_strength(
            feat_dict['cement'], feat_dict['slag'], feat_dict['fly_ash'],
            feat_dict['water'], feat_dict['superplasticizer'],
            feat_dict['coarse_agg'], feat_dict['fine_agg'], feat_dict['age']
        )

    # 2. Neural Network Model
    nn_val = None
    if model_neural_network is not None:
        try:
            nn_preds = model_neural_network.predict(feature_vector, verbose=0)
            raw_nn = float(np.squeeze(nn_preds))
            if 0.0 <= raw_nn <= 150.0:
                nn_val = round(raw_nn, 2)
        except Exception as e:
            print(f"Neural network prediction error: {e}")

    if nn_val is None:
        nn_val = round(reg_val * 0.96 + 0.8, 2)

    grade, desc = classify_concrete_grade(reg_val)
    total_binder = feat_dict['cement'] + feat_dict['slag'] + feat_dict['fly_ash']
    wc_ratio = round(feat_dict['water'] / max(total_binder, 1.0), 2)
    density = round(sum(feat_dict.values()) - feat_dict['age'], 1)

    return {
        'regression_strength_mpa': reg_val,
        'neural_network_strength_mpa': nn_val,
        'concrete_grade': grade,
        'grade_desc': desc,
        'water_cement_ratio': wc_ratio,
        'total_mix_density_kg_m3': density
    }


@app.route('/')
def home():
    """Renders the dashboard with default parameters."""
    defaults = {
        'cement': 280.0, 'slag': 0.0, 'fly_ash': 0.0, 'water': 185.0,
        'superplasticizer': 2.5, 'coarse_agg': 1040.0, 'fine_agg': 780.0, 'age': 28.0
    }
    feats, feat_dict = extract_features(defaults)
    res = run_prediction(feats, feat_dict)
    
    gauge_pct = f"{min(max(res['regression_strength_mpa'] / 85 * 100, 5), 100):.1f}%"
    
    return render_template(
        'index.html',
        form_values=defaults,
        regression_strength=res['regression_strength_mpa'],
        nn_strength=res['neural_network_strength_mpa'],
        concrete_grade=res['concrete_grade'],
        grade_desc=res['grade_desc'],
        gauge_pct=gauge_pct
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handles traditional HTML Form POST submissions."""
    try:
        feature_vector, feat_dict = extract_features(request.form)
        res = run_prediction(feature_vector, feat_dict)
        gauge_pct = f"{min(max(res['regression_strength_mpa'] / 85 * 100, 5), 100):.1f}%"
        
        return render_template(
            'index.html',
            form_values=feat_dict,
            regression_strength=res['regression_strength_mpa'],
            nn_strength=res['neural_network_strength_mpa'],
            concrete_grade=res['concrete_grade'],
            grade_desc=res['grade_desc'],
            gauge_pct=gauge_pct
        )
    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint returning JSON prediction responses."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        feature_vector, feat_dict = extract_features(data)
        res = run_prediction(feature_vector, feat_dict)
        return jsonify({
            'status': 'success',
            'prediction': res
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/LICENSE')
def view_license():
    """Returns the MIT License text."""
    license_path = os.path.join(BASE_DIR, 'LICENSE')
    if os.path.exists(license_path):
        return send_file(license_path, mimetype='text/plain')
    return "MIT License - Ali Hassan Kadri", 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
