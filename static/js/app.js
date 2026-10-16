/**
 * Concrete Compressive Strength Prediction
 * Frontend Controller & Dynamics
 * Author: Ali Hassan Kadri
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const form = document.getElementById('prediction-form');
  const predictBtn = document.getElementById('btn-predict');
  const spinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');
  
  // Real-time diagnostics
  const diagWc = document.getElementById('diag-wc');
  const diagDensity = document.getElementById('diag-density');
  const diagBinder = document.getElementById('diag-binder');
  
  // Result elements
  const resRegVal = document.getElementById('result-regression-val');
  const resNnVal = document.getElementById('result-nn-val');
  const resGrade = document.getElementById('result-grade-tag');
  const resDescription = document.getElementById('result-description');
  const gaugeFill = document.getElementById('gauge-fill-bar');

  // Input pairs (slider <-> number)
  const inputPairs = [
    { num: 'cement', slider: 'cement-slider' },
    { num: 'slag', slider: 'slag-slider' },
    { num: 'fly_ash', slider: 'fly_ash-slider' },
    { num: 'water', slider: 'water-slider' },
    { num: 'superplasticizer', slider: 'superplasticizer-slider' },
    { num: 'coarse_agg', slider: 'coarse_agg-slider' },
    { num: 'fine_agg', slider: 'fine_agg-slider' },
    { num: 'age', slider: 'age-slider' }
  ];

  // Sync Slider and Number inputs
  inputPairs.forEach(pair => {
    const numEl = document.getElementById(pair.num);
    const sliderEl = document.getElementById(pair.slider);

    if (numEl && sliderEl) {
      numEl.addEventListener('input', () => {
        sliderEl.value = numEl.value;
        updateDiagnostics();
      });

      sliderEl.addEventListener('input', () => {
        numEl.value = sliderEl.value;
        updateDiagnostics();
      });
    }
  });

  // Calculate live engineering diagnostics
  function updateDiagnostics() {
    const cement = parseFloat(document.getElementById('cement').value) || 0;
    const slag = parseFloat(document.getElementById('slag').value) || 0;
    const flyAsh = parseFloat(document.getElementById('fly_ash').value) || 0;
    const water = parseFloat(document.getElementById('water').value) || 0;
    const sp = parseFloat(document.getElementById('superplasticizer').value) || 0;
    const coarse = parseFloat(document.getElementById('coarse_agg').value) || 0;
    const fine = parseFloat(document.getElementById('fine_agg').value) || 0;

    const totalBinder = cement + slag + flyAsh;
    const totalDensity = cement + slag + flyAsh + water + sp + coarse + fine;
    const wcRatio = totalBinder > 0 ? (water / totalBinder) : 0;

    // Update Binder Display
    if (diagBinder) diagBinder.textContent = totalBinder.toFixed(1) + ' kg';
    
    // Update Density Display
    if (diagDensity) diagDensity.textContent = totalDensity.toFixed(0) + ' kg/m³';

    // Update W/C Ratio & Colors
    if (diagWc) {
      diagWc.textContent = wcRatio.toFixed(2);
      diagWc.className = 'diagnostic-value';
      if (wcRatio >= 0.35 && wcRatio <= 0.50) {
        diagWc.classList.add('optimal');
      } else if (wcRatio > 0.55 || wcRatio < 0.30) {
        diagWc.classList.add('warning');
      }
    }
  }

  // Preset Configurations
  const presets = {
    'c25-slab': {
      cement: 280, slag: 0, fly_ash: 0, water: 185,
      superplasticizer: 2.5, coarse_agg: 1040, fine_agg: 780, age: 28
    },
    'c50-high': {
      cement: 420, slag: 110, fly_ash: 0, water: 155,
      superplasticizer: 9.5, coarse_agg: 990, fine_agg: 740, age: 28
    },
    'eco-flyash': {
      cement: 240, slag: 0, fly_ash: 130, water: 165,
      superplasticizer: 5.0, coarse_agg: 1020, fine_agg: 790, age: 56
    },
    'early-7d': {
      cement: 460, slag: 80, fly_ash: 0, water: 160,
      superplasticizer: 11.0, coarse_agg: 980, fine_agg: 710, age: 7
    },
    'mass-dam': {
      cement: 180, slag: 150, fly_ash: 60, water: 175,
      superplasticizer: 3.0, coarse_agg: 1120, fine_agg: 800, age: 90
    }
  };

  // Preset Click Handling
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const presetKey = btn.getAttribute('data-preset');
      const data = presets[presetKey];
      if (data) {
        Object.keys(data).forEach(key => {
          const numEl = document.getElementById(key);
          const sliderEl = document.getElementById(key + '-slider');
          if (numEl) numEl.value = data[key];
          if (sliderEl) sliderEl.value = data[key];
        });
        updateDiagnostics();
        triggerPrediction();
      }
    });
  });

  // Concrete strength classification based on Eurocode EN 206 / ACI
  function getClassification(strength) {
    if (strength < 20) {
      return {
        grade: 'C12/15 - Low Strength Concrete',
        desc: 'Suitable for plain mass concrete, lean blinding, trench fill, and non-structural levelling beds.'
      };
    } else if (strength < 30) {
      return {
        grade: 'C20/25 - Standard Structural',
        desc: 'Standard commercial grade for domestic foundations, slabs, footings, and pavement.'
      };
    } else if (strength < 40) {
      return {
        grade: 'C30/37 - Reinforced Commercial',
        desc: 'Durable reinforced concrete for external paved areas, floor slabs, structural walls, and beams.'
      };
    } else if (strength < 55) {
      return {
        grade: 'C40/50 - High Performance',
        desc: 'High-strength structural grade for heavy commercial buildings, columns, and prestressed components.'
      };
    } else {
      return {
        grade: 'C60+ - Ultra High Strength',
        desc: 'Advanced structural concrete for high-rise infrastructure, marine piers, and critical civil works.'
      };
    }
  }

  // Update UI with predictions
  function renderResults(regStrength, nnStrength) {
    const regVal = parseFloat(regStrength) || 0;
    const nnVal = parseFloat(nnStrength) || 0;

    // Animate Number Count
    if (resRegVal) resRegVal.textContent = regVal.toFixed(2);
    if (resNnVal) resNnVal.textContent = nnVal.toFixed(2) + ' MPa';

    // Gauge bar width (capped at 100 MPa)
    const percentage = Math.min(Math.max((regVal / 85) * 100, 5), 100);
    if (gaugeFill) gaugeFill.style.width = percentage + '%';

    // Classification
    const classInfo = getClassification(regVal);
    if (resGrade) resGrade.textContent = classInfo.grade;
    if (resDescription) resDescription.textContent = classInfo.desc;
  }

  // Trigger Prediction via Fetch API
  function triggerPrediction() {
    if (!form) return;

    if (spinner) spinner.style.display = 'inline-block';
    if (btnText) btnText.textContent = 'Calculating...';
    if (predictBtn) predictBtn.disabled = true;

    const payload = {
      cement: parseFloat(document.getElementById('cement').value) || 0,
      slag: parseFloat(document.getElementById('slag').value) || 0,
      fly_ash: parseFloat(document.getElementById('fly_ash').value) || 0,
      water: parseFloat(document.getElementById('water').value) || 0,
      superplasticizer: parseFloat(document.getElementById('superplasticizer').value) || 0,
      coarse_agg: parseFloat(document.getElementById('coarse_agg').value) || 0,
      fine_agg: parseFloat(document.getElementById('fine_agg').value) || 0,
      age: parseFloat(document.getElementById('age').value) || 28
    };

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        renderResults(
          data.prediction.regression_strength_mpa,
          data.prediction.neural_network_strength_mpa
        );
      } else {
        alert(data.message || 'Error calculating prediction');
      }
    })
    .catch(err => {
      console.warn('AJAX prediction failed, fallback to native submission:', err);
      form.submit();
    })
    .finally(() => {
      if (spinner) spinner.style.display = 'none';
      if (btnText) btnText.textContent = 'Predict Compressive Strength';
      if (predictBtn) predictBtn.disabled = false;
    });
  }

  // Form submission intercept
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      triggerPrediction();
    });
  }

  // Initialize diagnostics on page load
  updateDiagnostics();

  // If initial server prediction values exist in hidden attributes, render gauge
  const initialReg = document.getElementById('initial-reg-value');
  const initialNn = document.getElementById('initial-nn-value');
  if (initialReg && initialReg.value) {
    renderResults(parseFloat(initialReg.value), parseFloat(initialNn ? initialNn.value : 0));
  }
});
