"""
Script to train and save PMEmo baseline models
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("🎯 PMEmo Baseline Models Training and Saving")
print("=" * 70)

# Set paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'PMEmo2019' / 'processed'
MODEL_DIR = BASE_DIR / 'models' / 'pmemo_baseline'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Data directory: {DATA_DIR}")
print(f"📂 Model directory: {MODEL_DIR}")

# Load feature data
print("\n" + "=" * 70)
print("📊 Loading Data")
print("=" * 70)

feature_file = DATA_DIR / 'pmemo_features_rich.csv'
if not feature_file.exists():
    print(f"❌ Feature file not found: {feature_file}")
    print("   Please run the PMEmo feature extraction notebook first!")
    exit(1)

df = pd.read_csv(feature_file)
print(f"✅ Loaded {len(df)} songs")
print(f"   Total columns: {len(df.columns)}")

# Prepare features and labels
feature_cols = [col for col in df.columns if col not in ['musicId', 'valence', 'arousal']]
X = df[feature_cols].values
y_valence = df['valence'].values
y_arousal = df['arousal'].values

print(f"\n📊 Data shape:")
print(f"   Features (X): {X.shape}")
print(f"   Valence (y): {y_valence.shape}, range: [{y_valence.min():.2f}, {y_valence.max():.2f}]")
print(f"   Arousal (y): {y_arousal.shape}, range: [{y_arousal.min():.2f}, {y_arousal.max():.2f}]")

# Split data (70% train / 15% val / 15% test)
print("\n" + "=" * 70)
print("✂️ Splitting Data (70% / 15% / 15%)")
print("=" * 70)

X_temp, X_test, y_val_temp, y_val_test, y_aro_temp, y_aro_test = train_test_split(
    X, y_valence, y_arousal, test_size=0.15, random_state=RANDOM_STATE
)

X_train, X_val, y_val_train, y_val_val, y_aro_train, y_aro_val = train_test_split(
    X_temp, y_val_temp, y_aro_temp, test_size=0.15/(1-0.15), random_state=RANDOM_STATE
)

print(f"   Train:      {X_train.shape[0]:4d} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Validation: {X_val.shape[0]:4d} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   Test:       {X_test.shape[0]:4d} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Standardize features
print("\n" + "=" * 70)
print("🔧 Standardizing Features")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features standardized (mean=0, std=1)")

# Train Linear Regression models
print("\n" + "=" * 70)
print("🤖 Training Linear Regression Models")
print("=" * 70)

lr_valence = LinearRegression()
lr_valence.fit(X_train_scaled, y_val_train)
lr_val_pred = lr_valence.predict(X_val_scaled)
lr_val_r2 = r2_score(y_val_val, lr_val_pred)
lr_val_rmse = np.sqrt(mean_squared_error(y_val_val, lr_val_pred))

print(f"✅ Valence: R² = {lr_val_r2:.4f}, RMSE = {lr_val_rmse:.4f}")

lr_arousal = LinearRegression()
lr_arousal.fit(X_train_scaled, y_aro_train)
lr_aro_pred = lr_arousal.predict(X_val_scaled)
lr_aro_r2 = r2_score(y_aro_val, lr_aro_pred)
lr_aro_rmse = np.sqrt(mean_squared_error(y_aro_val, lr_aro_pred))

print(f"✅ Arousal: R² = {lr_aro_r2:.4f}, RMSE = {lr_aro_rmse:.4f}")

# Train Random Forest models
print("\n" + "=" * 70)
print("🤖 Training Random Forest Models")
print("=" * 70)

rf_valence = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_valence.fit(X_train_scaled, y_val_train)
rf_val_pred = rf_valence.predict(X_val_scaled)
rf_val_r2 = r2_score(y_val_val, rf_val_pred)
rf_val_rmse = np.sqrt(mean_squared_error(y_val_val, rf_val_pred))

print(f"✅ Valence: R² = {rf_val_r2:.4f}, RMSE = {rf_val_rmse:.4f}")

rf_arousal = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_arousal.fit(X_train_scaled, y_aro_train)
rf_aro_pred = rf_arousal.predict(X_val_scaled)
rf_aro_r2 = r2_score(y_aro_val, rf_aro_pred)
rf_aro_rmse = np.sqrt(mean_squared_error(y_aro_val, rf_aro_pred))

print(f"✅ Arousal: R² = {rf_aro_r2:.4f}, RMSE = {rf_aro_rmse:.4f}")

# Train SVR models (paper method)
print("\n" + "=" * 70)
print("🤖 Training SVR Models (Paper Method)")
print("=" * 70)

svr_valence = SVR(kernel='rbf', gamma='scale')
svr_valence.fit(X_train_scaled, y_val_train)
svr_val_pred = svr_valence.predict(X_val_scaled)
svr_val_r2 = r2_score(y_val_val, svr_val_pred)
svr_val_rmse = np.sqrt(mean_squared_error(y_val_val, svr_val_pred))

print(f"✅ Valence: R² = {svr_val_r2:.4f}, RMSE = {svr_val_rmse:.4f}")

svr_arousal = SVR(kernel='rbf', gamma='scale')
svr_arousal.fit(X_train_scaled, y_aro_train)
svr_aro_pred = svr_arousal.predict(X_val_scaled)
svr_aro_r2 = r2_score(y_aro_val, svr_aro_pred)
svr_aro_rmse = np.sqrt(mean_squared_error(y_aro_val, svr_aro_pred))

print(f"✅ Arousal: R² = {svr_aro_r2:.4f}, RMSE = {svr_aro_rmse:.4f}")

# Save all models
print("\n" + "=" * 70)
print("💾 Saving Models")
print("=" * 70)

models_to_save = [
    (lr_valence, 'linear_regression_valence.pkl'),
    (lr_arousal, 'linear_regression_arousal.pkl'),
    (rf_valence, 'random_forest_valence.pkl'),
    (rf_arousal, 'random_forest_arousal.pkl'),
    (svr_valence, 'svr_valence.pkl'),
    (svr_arousal, 'svr_arousal.pkl'),
    (scaler, 'feature_scaler.pkl'),
]

for model, filename in models_to_save:
    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)
    file_size = model_path.stat().st_size / 1024 / 1024  # MB
    print(f"   ✅ {filename:<35} ({file_size:>6.2f} MB)")

# Save performance summary
print("\n" + "=" * 70)
print("📊 Performance Summary")
print("=" * 70)

summary = pd.DataFrame({
    'Model': ['Linear Regression', 'Linear Regression', 'Random Forest', 'Random Forest', 'SVR', 'SVR'],
    'Target': ['Valence', 'Arousal', 'Valence', 'Arousal', 'Valence', 'Arousal'],
    'R²': [lr_val_r2, lr_aro_r2, rf_val_r2, rf_aro_r2, svr_val_r2, svr_aro_r2],
    'RMSE': [lr_val_rmse, lr_aro_rmse, rf_val_rmse, rf_aro_rmse, svr_val_rmse, svr_aro_rmse]
})

print(summary.to_string(index=False))

# Save summary to CSV
summary_path = MODEL_DIR / 'performance_summary.csv'
summary.to_csv(summary_path, index=False)
print(f"\n💾 Performance summary saved: {summary_path}")

print("\n" + "=" * 70)
print("✅ All PMEmo models saved successfully!")
print("=" * 70)

