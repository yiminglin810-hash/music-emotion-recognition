# 🎵 Music Emotion Recognition - Final Year Project

> **Author**: Yiming Lin  
> **Supervisor**: Jacopo  
> **Institution**: University of Liverpool  
> **Academic Year**: 2024-2025 (Semester 1)

A comprehensive study on **Music Emotion Recognition (MER)** using **traditional machine learning** approaches with **hand-crafted audio features**.

---

## 📋 Project Overview

This project explores emotion recognition in music using the **Valence-Arousal (V-A) dimensional model** across three major datasets:

- **DEAM**: 1,802 songs with crowd-sourced annotations
- **MuSe**: 90,001 music clips with dynamic temporal annotations
- **PMEmo**: 794 songs with static + dynamic annotations + physiological signals (EDA)

### 🎯 Research Goals

1. **Establish baselines** using traditional ML models (Linear Regression, Random Forest, SVR)
2. **Compare feature extraction methods**: librosa (334-dim Rich Stats) vs OpenSMILE (6373-dim ComParE)
3. **Investigate shortcutting phenomenon**: Why do models cluster predictions around the mean?
4. **Replicate state-of-the-art**: Reproduce PMEmo paper's methodology

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
# Clone the repository
git clone https://github.com/yiminglin810-hash/music-emotion-recognition.git
cd music-emotion-recognition

# Install dependencies
pip install -r requirements.txt

# Verify environment
python verify_environment.py
```

**System Requirements:**
- Python 3.8+
- CUDA 11.8 (for future deep learning experiments)
- 16GB RAM (recommended)
- 10GB disk space (excluding datasets)

---

### 2️⃣ Download Datasets

See [`data/README.md`](data/README.md) for detailed instructions.

**Quick Links:**
- **DEAM**: https://cvml.unige.ch/databases/DEAM/
- **MuSe**: https://zenodo.org/record/5090631
- **PMEmo**: https://github.com/HuiZhangDB/PMEmo

---

### 3️⃣ Run Experiments

**Recommended execution order:**

```bash
# DEAM Dataset (Baseline)
jupyter notebook notebooks/notebooks_en/01_deam_data_loader_EN.ipynb
jupyter notebook notebooks/notebooks_en/02_deam_feature_extraction_EN.ipynb
jupyter notebook notebooks/notebooks_en/03_deam_baseline_models_EN.ipynb

# MuSe Dataset
jupyter notebook notebooks/notebooks_en/04_muse_data_analysis_EN.ipynb
jupyter notebook notebooks/notebooks_en/05_muse_feature_extraction_EN.ipynb
jupyter notebook notebooks/notebooks_en/06_muse_baseline_models_EN.ipynb

# PMEmo Dataset (Main Focus)
jupyter notebook notebooks/notebooks_en/07_pmemo_baseline_models_EN.ipynb  # ⭐ Key notebook
```

> 💡 **Note**: Chinese versions are available in `notebooks/notebooks_cn/` for reference.

---

## 📊 Key Results

### PMEmo Dataset - Baseline Models (10-Fold Cross-Validation)

| Model | Valence RMSE | Arousal RMSE | Valence r | Arousal r |
|-------|--------------|--------------|-----------|-----------|
| **Linear Regression** | 0.2717±0.293 | 0.1887±0.108 | 0.324 | 0.574 |
| **Random Forest** | **0.1166±0.013** | **0.1073±0.005** | **0.698** | **0.814** |
| **SVR (RBF)** | 0.1185±0.012 | 0.1158±0.006 | 0.689 | 0.788 |

### Comparison with PMEmo Paper

| Method | Features | Val RMSE | Aro RMSE | Val r | Aro r |
|--------|----------|----------|----------|-------|-------|
| **PMEmo Paper (SVR)** | 6373-dim ComParE | 0.124 | 0.102 | 0.638 | 0.764 |
| **Our Method (RF)** | 334-dim Rich Stats | **0.117** | 0.107 | **0.698** | **0.814** |

**🎉 Key Finding**: Random Forest with Rich Statistical Features achieves **competitive or better performance** than the original paper using 19x fewer features!

---

## 🔬 Methodology

### Feature Extraction Pipeline

**Step 1: Audio Loading**
- Tool: `librosa`
- Sample Rate: 22,050 Hz
- Duration: Full song (DEAM/PMEmo) or 30s clips (MuSe)

**Step 2: Low-Level Descriptor (LLD) Extraction**
- **MFCC** (13 coefficients): Timbral texture
- **Chroma** (12 bins): Pitch content
- **Spectral Centroid, Bandwidth, Rolloff**: Frequency distribution
- **Zero-Crossing Rate**: Noisiness
- **RMS Energy**: Loudness

**Step 3: Statistical Aggregation (334-dim)**

For each LLD, compute **9 statistics**:
- Central Tendency: Mean, Median
- Dispersion: Std, Q25, Q75
- Range: Min, Max
- Distribution Shape: Skewness, Kurtosis

**Rationale**: Capture full distribution shape, not just mean/std (as in PMEmo paper)

---

### Models

| Model | Algorithm | Key Parameters |
|-------|-----------|----------------|
| **Linear Regression** | OLS | - |
| **Random Forest** | Ensemble Trees | `n_estimators=100`, `max_depth=20` |
| **SVR** | RBF Kernel | `C=1.0`, `gamma='scale'` |

**Evaluation**:
- **Metrics**: RMSE, NRMSE, MAE, R², Pearson r
- **Method**: 10-fold cross-validation (small datasets: PMEmo, DEAM)
- **Baseline**: 70/15/15 train/val/test split (large dataset: MuSe)

---

## 🤔 Key Findings & Open Questions

### 1️⃣ Shortcutting Phenomenon

**Observation**: All models (LR, RF, SVR) show **prediction clustering around the mean**, failing to capture emotional extremes (very happy/sad music).

**Evidence**:
- High Pearson r (0.698-0.814) but limited prediction range
- See `docs/figures/pmemo_three_models_comparison.png`

**Question**: Is high correlation misleading if models can't predict extremes? 🤔

---

### 2️⃣ Feature Quality vs Annotation Quality

**Our Method**:
- 334-dim librosa features
- RMSE: 0.117 (Valence), 0.107 (Arousal)

**PMEmo Paper**:
- 6373-dim OpenSMILE ComParE
- RMSE: 0.124 (Valence), 0.102 (Arousal)

**Question**: Why do fewer features achieve comparable performance? 🤔
- Are Rich Stats (9 statistics) more informative than ComParE (mean/std)?
- Is annotation quality the bottleneck (crowd-sourced vs expert)?

---

### 3️⃣ Model Selection

**Expectation (from literature)**: SVR should outperform RF for small datasets

**Reality**: RF > SVR in our experiments

**Question**: Why does RF perform better? 🤔
- Overfitting in SVR?
- Suboptimal hyperparameters?
- Feature engineering better suited for tree-based models?

---

## 📁 Project Structure

```
Semester1/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
│
├── notebooks/                   # Jupyter experiments ⭐
│   ├── notebooks_en/            # English versions (main)
│   │   ├── 01_deam_data_loader_EN.ipynb
│   │   ├── 02_deam_feature_extraction_EN.ipynb
│   │   ├── 03_deam_baseline_models_EN.ipynb
│   │   ├── 04_muse_data_analysis_EN.ipynb
│   │   ├── 05_muse_feature_extraction_EN.ipynb
│   │   ├── 06_muse_baseline_models_EN.ipynb
│   │   └── 07_pmemo_baseline_models_EN.ipynb
│   └── notebooks_cn/            # Chinese versions (reference)
│       └── [Same notebooks in Chinese]
│
├── src/                         # Reusable Python modules
│   ├── features/                # Feature extraction
│   ├── models/                  # Model training
│   ├── evaluation/              # Evaluation metrics
│   └── utils/                   # Data loading, file I/O
│
├── scripts/                     # Standalone scripts
│   ├── extract_all_features.py  # Batch feature extraction
│   └── create_muse_baseline.py  # MuSe baseline training
│
├── data/                        # Datasets (not tracked in Git)
│   ├── DEAM/                    # Download separately
│   ├── MuSe/                    # Download separately
│   └── PMEmo2019/               # Download separately
│
├── models/                      # Trained models (not tracked in Git)
│   └── README.md                # How to generate models
│
└── docs/                        # Reports and documentation
    ├── PMEmo_Experiment_Results_Simple.tex  # Main report ⭐
    ├── Week2_DEAM_Baseline_Report.tex
    ├── Week2_MuSe_Experiment_Report.tex
    ├── LaTeX编译指南.md
    └── figures/                 # Experiment visualizations
        ├── pmemo_three_models_comparison.png
        └── pmemo_valence_arousal_2d_overlay.png
```

---

## 📄 Reports

### Main Report (LaTeX)

See [`docs/PMEmo_Experiment_Results_Simple.tex`](docs/PMEmo_Experiment_Results_Simple.tex) for:
- Detailed methodology
- Full experimental results
- Statistical comparisons
- Discussion questions for supervisor

**To compile:**
```bash
cd docs
pdflatex PMEmo_Experiment_Results_Simple.tex
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Audio Processing** | librosa, soundfile, audioread |
| **Feature Extraction** | librosa (current), OpenSMILE (planned) |
| **Machine Learning** | scikit-learn (LR, RF, SVR) |
| **Deep Learning** | PyTorch (future work) |
| **Data Analysis** | pandas, numpy, scipy |
| **Visualization** | matplotlib, seaborn |
| **Reporting** | Jupyter Notebook, LaTeX |

---

## 🔮 Future Work

### Short-term (Next Semester)

1. **Address Shortcutting**
   - Loss function engineering (weighted MSE for extremes)
   - Data augmentation for extreme emotions
   - Post-processing prediction adjustment

2. **Feature Engineering**
   - Implement OpenSMILE ComParE for fair comparison
   - Explore other feature sets (eGeMAPS, MFCC deltas)
   - Feature selection (PCA, mutual information)

3. **Hyperparameter Optimization**
   - Grid search for SVR (C, gamma)
   - Random search for RF (n_estimators, max_depth)

### Long-term

4. **Deep Learning Approaches**
   - CNN on spectrograms (end-to-end learning)
   - Transformer-based models (Music Transformer)
   - Pre-trained embeddings (CLAP, Jukebox)

5. **Multimodal Fusion** (PMEmo only)
   - Audio + Lyrics (text embeddings)
   - Audio + EDA signals (physiological arousal)

6. **Dynamic MER** (PMEmo, MuSe)
   - Temporal models (LSTM, GRU)
   - Sequence-to-sequence prediction

---

## 📚 References

### Datasets
1. **DEAM**: Aljanaki et al. (2017) - "Developing a benchmark for emotional analysis of music"
2. **MuSe**: Deng et al. (2021) - "MuSe: A Multimodal Dataset of Stressed Emotion"
3. **PMEmo**: Zhang et al. (2018) - "The PMEmo Dataset for Music Emotion Recognition"

### Feature Sets
- **ComParE**: Schuller et al. (2013) - "The INTERSPEECH 2013 ComParE challenge"
- **eGeMAPS**: Eyben et al. (2016) - "The Geneva Minimalistic Acoustic Parameter Set"

### Related Work
- Russell (1980) - "A circumplex model of affect" (V-A model)
- Panda et al. (2018) - "Audio features for music emotion recognition" (survey)

---

## 📧 Contact

**Student**: Yiming Lin  
**Supervisor**: Jacopo  
**Institution**: University of Liverpool  
**Email**: sgylin19@liverpool.ac.uk  
**GitHub**: https://github.com/yiminglin810-hash  
**Portfolio**: https://yiminglin-portfolio.netlify.app

---

## 📝 License

This project is for academic purposes only. Datasets are subject to their respective licenses.

---

## 🙏 Acknowledgments

- Thanks to **Jacopo** for supervision and guidance
- Thanks to the creators of **DEAM**, **MuSe**, and **PMEmo** datasets
- Thanks to the open-source community (librosa, scikit-learn, PyTorch)

---

**Last Updated**: 2025-12-03

