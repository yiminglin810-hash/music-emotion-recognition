# 🤖 Trained Models

This directory contains **trained machine learning models** for Music Emotion Recognition. **Model files are NOT tracked in Git** due to file size (50+ MB total).

---

## 🗂️ Directory Structure

```
models/
├── README.md                           # This file
│
├── muse_baseline/                      # MuSe baseline models (12 files, ~30 MB)
│   ├── lr_valence_train.pkl            # Linear Regression (Valence, Train)
│   ├── lr_valence_val.pkl              # Linear Regression (Valence, Validation)
│   ├── lr_valence_test.pkl             # Linear Regression (Valence, Test)
│   ├── lr_arousal_train.pkl            # Linear Regression (Arousal, Train)
│   ├── lr_arousal_val.pkl              # Linear Regression (Arousal, Validation)
│   ├── lr_arousal_test.pkl             # Linear Regression (Arousal, Test)
│   ├── rf_valence_train.pkl            # Random Forest (Valence, Train)
│   ├── rf_valence_val.pkl              # Random Forest (Valence, Validation)
│   ├── rf_valence_test.pkl             # Random Forest (Valence, Test)
│   ├── rf_arousal_train.pkl            # Random Forest (Arousal, Train)
│   ├── rf_arousal_val.pkl              # Random Forest (Arousal, Validation)
│   └── rf_arousal_test.pkl             # Random Forest (Arousal, Test)
│
├── muse_baseline_133dim/               # MuSe upgraded models (8 files, ~20 MB)
│   ├── lr_valence.pkl
│   ├── lr_arousal.pkl
│   ├── rf_valence.pkl
│   ├── rf_arousal.pkl
│   ├── scaler_valence.pkl              # StandardScaler for Valence
│   ├── scaler_arousal.pkl              # StandardScaler for Arousal
│   ├── feature_importance_valence.csv  # RF feature importance
│   └── feature_importance_arousal.csv
│
└── pmemo_baseline/                     # PMEmo baseline models (NOT saved yet)
    └── (Models trained but not persisted in notebook 06)
```

**Status:**
- ✅ DEAM: Models trained in notebook 03 (not persisted to disk)
- ✅ MuSe: Models saved (12 + 8 = 20 .pkl files)
- ⚠️ PMEmo: Models trained in notebook 06 (not persisted to disk)

---

## 🚀 How to Generate Models

Models are trained directly in Jupyter Notebooks. Follow these steps:

### 1️⃣ DEAM Baseline Models

**Notebook**: `notebooks/03_baseline_models.ipynb`

**Models**:
- Linear Regression (LR)
- Random Forest (RF)

**Run**:
```bash
jupyter notebook notebooks/03_baseline_models.ipynb
# Execute all cells
```

**Output**: Models stored in memory (not saved to disk)

---

### 2️⃣ MuSe Baseline Models

**Notebook**: `notebooks/05_muse_baseline_models.ipynb`

**Models**:
- Linear Regression (LR): ~0.5 MB per model
- Random Forest (RF): ~10 MB per model

**Run**:
```bash
jupyter notebook notebooks/05_muse_baseline_models.ipynb
# Execute all cells
```

**Output**: Models saved to `models/muse_baseline/` and `models/muse_baseline_133dim/`

---

### 3️⃣ PMEmo Baseline Models

**Notebook**: `notebooks/06_pmemo_baseline_models.ipynb`

**Models**:
- Linear Regression (LR)
- Random Forest (RF)
- Support Vector Regression (SVR)

**Run**:
```bash
jupyter notebook notebooks/06_pmemo_baseline_models.ipynb
# Execute all cells (takes ~15 minutes for 10-fold CV)
```

**Output**: Models trained but **not saved** (memory only)

**To save models**, add to notebook (after Cell 31):
```python
import joblib

# Save SVR models
joblib.dump(svr_val, 'models/pmemo_baseline/svr_valence.pkl')
joblib.dump(svr_aro, 'models/pmemo_baseline/svr_arousal.pkl')

# Save RF models
joblib.dump(rf_val, 'models/pmemo_baseline/rf_valence.pkl')
joblib.dump(rf_aro, 'models/pmemo_baseline/rf_arousal.pkl')

# Save scalers
joblib.dump(scaler, 'models/pmemo_baseline/scaler.pkl')
```

---

## 📊 Model Performance Summary

### DEAM Dataset (1,802 songs)

| Model | Valence R² | Arousal R² | Valence RMSE | Arousal RMSE |
|-------|-----------|-----------|--------------|--------------|
| **LR** | ~0.30 | ~0.25 | ~1.2 | ~1.0 |
| **RF** | ~0.45 | ~0.40 | ~1.0 | ~0.85 |

**Split**: 70/15/15 (train/val/test)

---

### MuSe Dataset (90,001 clips)

| Model | Valence R² | Arousal R² | Valence RMSE | Arousal RMSE |
|-------|-----------|-----------|--------------|--------------|
| **LR (88-dim)** | 0.182 | 0.201 | 0.3578 | 0.3404 |
| **RF (88-dim)** | 0.438 | 0.432 | 0.2965 | 0.2871 |
| **LR (133-dim)** | 0.193 | 0.215 | 0.3553 | 0.3375 |
| **RF (133-dim)** | **0.451** | **0.441** | **0.2931** | **0.2847** |

**Split**: 70/15/15 (train/val/test)  
**Improvement**: +3-4% R² with feature upgrade (88→133 dim)

---

### PMEmo Dataset (794 songs)

| Model | Valence RMSE | Arousal RMSE | Valence r | Arousal r |
|-------|--------------|--------------|-----------|-----------|
| **LR** | 0.272±0.293 | 0.189±0.108 | 0.324 | 0.574 |
| **RF** | **0.117±0.013** | **0.107±0.005** | **0.698** | **0.814** |
| **SVR** | 0.119±0.012 | 0.116±0.006 | 0.689 | 0.788 |

**Evaluation**: 10-fold cross-validation  
**Best Model**: Random Forest (RF)

---

## 🔧 Model Loading Example

### Load MuSe Models

```python
import joblib
import pandas as pd
import numpy as np

# Load trained models
rf_val = joblib.load('models/muse_baseline_133dim/rf_valence.pkl')
rf_aro = joblib.load('models/muse_baseline_133dim/rf_arousal.pkl')

# Load scalers
scaler_val = joblib.load('models/muse_baseline_133dim/scaler_valence.pkl')
scaler_aro = joblib.load('models/muse_baseline_133dim/scaler_arousal.pkl')

# Load test features
X_test = pd.read_csv('data/MuSe/processed/muse_features_133dim_test.csv')

# Standardize features
X_test_scaled_val = scaler_val.transform(X_test)
X_test_scaled_aro = scaler_aro.transform(X_test)

# Predict
y_pred_val = rf_val.predict(X_test_scaled_val)
y_pred_aro = rf_aro.predict(X_test_scaled_aro)

print(f"Valence predictions: {y_pred_val[:5]}")
print(f"Arousal predictions: {y_pred_aro[:5]}")
```

---

## 📏 Model File Sizes

| Model Type | Valence | Arousal | Total |
|------------|---------|---------|-------|
| **Linear Regression** | ~0.5 MB | ~0.5 MB | ~1 MB |
| **Random Forest** | ~10 MB | ~10 MB | ~20 MB |
| **SVR** (PMEmo) | ~2 MB | ~2 MB | ~4 MB |

**Total (all models)**: ~50 MB

---

## 🔬 Model Hyperparameters

### Linear Regression (scikit-learn)

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression(
    fit_intercept=True,
    normalize=False,
    n_jobs=-1
)
```

**Notes**: No hyperparameters to tune

---

### Random Forest (scikit-learn)

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,          # Number of trees
    max_depth=20,              # Max tree depth
    min_samples_split=10,      # Min samples to split node
    min_samples_leaf=5,        # Min samples in leaf
    random_state=42,           # Reproducibility
    n_jobs=-1                  # Use all CPU cores
)
```

**Notes**:
- DEAM/MuSe: `n_estimators=100`, `max_depth=20`
- PMEmo (10-fold CV): `n_estimators=50`, `max_depth=10` (for speed)

---

### Support Vector Regression (scikit-learn)

```python
from sklearn.svm import SVR

svr = SVR(
    kernel='rbf',              # Radial basis function
    C=1.0,                     # Regularization parameter
    gamma='scale',             # Kernel coefficient (auto)
    epsilon=0.1                # Epsilon-tube width
)
```

**Notes**: Used only for PMEmo (following paper methodology)

---

## 🚨 Important Notes

### Why Models Are NOT Tracked in Git

1. **File Size**: Total ~50 MB exceeds GitHub's recommended limit (10 MB per file)
2. **Reproducibility**: Models can be regenerated from notebooks (deterministic with `random_state`)
3. **Versioning**: Model weights change frequently during experimentation

### Alternatives for Model Sharing

1. **Git LFS** (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes
   git add models/*.pkl
   git commit -m "Add trained models via LFS"
   ```

2. **Cloud Storage** (Google Drive, OneDrive):
   - Upload `models/` directory
   - Share download link in README

3. **Hugging Face Model Hub**:
   - Upload models to https://huggingface.co/models
   - Version control + automatic hosting

---

## 🐛 Troubleshooting

### Issue 1: Model File Not Found

**Error**: `FileNotFoundError: models/muse_baseline/rf_valence.pkl`

**Solution**:
1. Run the corresponding notebook to generate models:
   ```bash
   jupyter notebook notebooks/05_muse_baseline_models.ipynb
   ```
2. Check that models are saved (look for `joblib.dump()` in notebook)

---

### Issue 2: Model Version Mismatch

**Error**: `ModuleNotFoundError: No module named 'sklearn.tree._tree'`

**Solution**:
1. Check scikit-learn version:
   ```bash
   pip show scikit-learn
   ```
2. Reinstall matching version:
   ```bash
   pip install scikit-learn==1.3.0
   ```
3. Retrain models with current version

---

### Issue 3: Memory Error When Loading RF

**Error**: `MemoryError` when loading `rf_arousal.pkl`

**Solution**:
1. Use lighter models (reduce `n_estimators` or `max_depth`)
2. Load models one at a time (not all together)
3. Increase system RAM or use cloud compute (Colab, Kaggle)

---

## 📚 References

### Model Documentation

- [scikit-learn: Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [scikit-learn: Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn: SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

### Persistence
- [joblib Documentation](https://joblib.readthedocs.io/en/latest/persistence.html)

---

## 🔗 Quick Commands

```bash
# Check model directory size
du -sh models/

# List all model files
find models/ -name "*.pkl" -type f

# Remove all models (to regenerate)
rm -rf models/muse_baseline/*.pkl
rm -rf models/muse_baseline_133dim/*.pkl
```

---

**Last Updated**: 2025-12-03

