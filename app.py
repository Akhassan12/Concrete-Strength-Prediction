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

# Suppress TensorFlow verbose logging & oneDNN notices; force CPU mode for cloud containers
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
nn_scaler = None

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
            nn_data = pickle.load(f)
        # Support both bundled (dict with model+scaler) and legacy (bare model) formats
        if isinstance(nn_data, dict) and 'model' in nn_data:
            model_neural_network = nn_data['model']
            nn_scaler = nn_data.get('scaler', None)
            print("[INFO] Deep Neural Network model loaded (with scaler).")
        else:
            model_neural_network = nn_data
            print("[INFO] Deep Neural Network model loaded (legacy format, no scaler).")
except Exception as e:
    print(f"[WARNING] Warning loading NeuralNetwork.pkl: {e}")


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


def validate_inputs(feat_dict):
    """
    Validates input mix features against physical laws and civil engineering standards.
    Returns:
        (errors: list of fatal validation error strings,
         advisories: list of warning strings for domain extrapolation)
    """
    errors = []
    advisories = []

    # 1. Non-negativity constraints
    for k, v in feat_dict.items():
        if k != 'age' and v < 0:
            errors.append(f"{k.replace('_', ' ').capitalize()} cannot be negative ({v} kg/m³).")
    
    if feat_dict['age'] <= 0:
        errors.append(f"Curing age must be strictly positive (> 0 days, got {feat_dict['age']}).")

    # 2. Minimum physical viability
    total_binder = feat_dict['cement'] + feat_dict['slag'] + feat_dict['fly_ash']
    if total_binder <= 0:
        errors.append("Total cementitious binder (cement + slag + fly ash) must be greater than 0 kg/m³.")
    
    if feat_dict['water'] <= 0:
        errors.append("Water content must be greater than 0 kg/m³.")

    # 3. Batch density plausibility (sum of ingredients per m³)
    total_density = sum(feat_dict.values()) - feat_dict['age']
    if total_density < 1400 or total_density > 3200:
        errors.append(f"Mix batch density ({total_density:.1f} kg/m³) is physically unviable for concrete (plausible: 1800–2800 kg/m³).")

    # 4. Domain extrapolation advisories (UCI dataset empirical bounds)
    if feat_dict['cement'] < 100 or feat_dict['cement'] > 540:
        advisories.append(f"Cement content ({feat_dict['cement']:.1f} kg/m³) is outside empirical training bounds [102, 540] kg/m³.")
    if feat_dict['age'] > 365:
        advisories.append(f"Curing age ({feat_dict['age']:.0f} days) exceeds max dataset age (365 days); predictions extrapolate.")
    wb_ratio = feat_dict['water'] / max(total_binder, 1.0)
    if wb_ratio < 0.22 or wb_ratio > 1.15:
        advisories.append(f"Water/binder ratio ({wb_ratio:.2f}) is outside typical concrete mix envelope [0.25, 0.85].")
    if feat_dict['superplasticizer'] > 32.2:
        advisories.append(f"Superplasticizer ({feat_dict['superplasticizer']:.1f} kg/m³) exceeds empirical max dosage (32.2 kg/m³).")

    return errors, advisories


def extract_features(data_dict):
    """Extracts features in exact sequence required by the models with strict validation."""
    try:
        cement = float(data_dict.get('cement', data_dict.get('Quanity of Cement in kg', 280.0)))
        slag = float(data_dict.get('slag', data_dict.get('Quanity of blast furnace slag in kg', 0.0)))
        fly_ash = float(data_dict.get('fly_ash', data_dict.get('Quanity of fly ash in kg', 0.0)))
        water = float(data_dict.get('water', data_dict.get('Quanity of water in kg', 185.0)))
        superplasticizer = float(data_dict.get('superplasticizer', data_dict.get('Quanity of superplasticizer in kg', 2.5)))
        coarse_agg = float(data_dict.get('coarse_agg', data_dict.get('Quanity of coarse_aggregate in kg', 1040.0)))
        fine_agg = float(data_dict.get('fine_agg', data_dict.get('Quanity of fine_aggregate in kg', 780.0)))
        age = float(data_dict.get('age', data_dict.get('No of days to cure', 28.0)))
        
        feat_dict = {
            'cement': cement, 'slag': slag, 'fly_ash': fly_ash, 'water': water,
            'superplasticizer': superplasticizer, 'coarse_agg': coarse_agg,
            'fine_agg': fine_agg, 'age': age
        }

        # Perform input validation
        errors, advisories = validate_inputs(feat_dict)
        if errors:
            raise ValueError("Input validation failed: " + "; ".join(errors))

        feature_vector = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
        return feature_vector, feat_dict, advisories
    except Exception as e:
        raise ValueError(f"{e}")


def run_prediction(feature_vector, feat_dict, advisories=None):
    """Executes predictions across both models. Returns errors if a model is unavailable."""
    # 1. Regression Model
    reg_val = None
    reg_error = None
    if model_regression is not None:
        try:
            preds = model_regression.predict(feature_vector)
            val = float(preds[0])
            reg_val = round(val, 2)
        except Exception as e:
            reg_error = f"Ensemble prediction failed: {e}"
            print(f"Regression prediction error: {e}")
    else:
        reg_error = "Ensemble model not loaded"

    # 2. Neural Network Model
    nn_val = None
    nn_error = None
    if model_neural_network is not None:
        try:
            # Apply scaler if available (required for correct NN predictions)
            if nn_scaler is not None:
                nn_input = nn_scaler.transform(feature_vector)
            else:
                nn_input = feature_vector
            nn_preds = model_neural_network.predict(nn_input, verbose=0)
            raw_nn = float(np.squeeze(nn_preds))
            nn_val = round(raw_nn, 2)
        except Exception as e:
            nn_error = f"Neural network prediction failed: {e}"
            print(f"Neural network prediction error: {e}")
    else:
        nn_error = "Neural network model not loaded"

    # Use whichever model succeeded for classification; prefer ensemble
    primary_strength = reg_val if reg_val is not None else nn_val
    if primary_strength is not None:
        grade, desc = classify_concrete_grade(primary_strength)
    else:
        grade = "N/A"
        desc = "Both models failed to produce a prediction."

    # Water-to-binder ratio (cement + slag + fly ash), not just w/c
    total_binder = feat_dict['cement'] + feat_dict['slag'] + feat_dict['fly_ash']
    wb_ratio = round(feat_dict['water'] / max(total_binder, 1.0), 2)
    density = round(sum(feat_dict.values()) - feat_dict['age'], 1)

    result = {
        'regression_strength_mpa': reg_val,
        'neural_network_strength_mpa': nn_val,
        'concrete_grade': grade,
        'grade_desc': desc,
        'water_binder_ratio': wb_ratio,
        'total_mix_density_kg_m3': density,
        'advisories': advisories or []
    }

    # Include error messages when models fail so the user knows
    if reg_error:
        result['regression_error'] = reg_error
    if nn_error:
        result['neural_network_error'] = nn_error

    return result


@app.route('/')
def home():
    """Renders the dashboard with default parameters."""
    defaults = {
        'cement': 280.0, 'slag': 0.0, 'fly_ash': 0.0, 'water': 185.0,
        'superplasticizer': 2.5, 'coarse_agg': 1040.0, 'fine_agg': 780.0, 'age': 28.0
    }
    feats, feat_dict, advisories = extract_features(defaults)
    res = run_prediction(feats, feat_dict, advisories)
    
    reg_strength = res['regression_strength_mpa']
    if reg_strength is not None:
        gauge_pct = f"{min(max(reg_strength / 85 * 100, 5), 100):.1f}%"
    else:
        gauge_pct = "5%"
    
    return render_template(
        'index.html',
        form_values=defaults,
        regression_strength=reg_strength if reg_strength is not None else 'N/A',
        nn_strength=res['neural_network_strength_mpa'] if res['neural_network_strength_mpa'] is not None else 'N/A',
        concrete_grade=res['concrete_grade'],
        grade_desc=res['grade_desc'],
        gauge_pct=gauge_pct,
        advisories=res.get('advisories', [])
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handles traditional HTML Form POST submissions."""
    try:
        feature_vector, feat_dict, advisories = extract_features(request.form)
        res = run_prediction(feature_vector, feat_dict, advisories)

        reg_strength = res['regression_strength_mpa']
        if reg_strength is not None:
            gauge_pct = f"{min(max(reg_strength / 85 * 100, 5), 100):.1f}%"
        else:
            gauge_pct = "5%"
        
        return render_template(
            'index.html',
            form_values=feat_dict,
            regression_strength=reg_strength if reg_strength is not None else 'N/A',
            nn_strength=res['neural_network_strength_mpa'] if res['neural_network_strength_mpa'] is not None else 'N/A',
            concrete_grade=res['concrete_grade'],
            grade_desc=res['grade_desc'],
            gauge_pct=gauge_pct,
            advisories=res.get('advisories', [])
        )
    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint returning JSON prediction responses."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        feature_vector, feat_dict, advisories = extract_features(data)
        res = run_prediction(feature_vector, feat_dict, advisories)
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
