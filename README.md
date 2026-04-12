<div align="center">

# Accuracy is not certainty: 
# code for uncertainty-aware mangrove mapping to inform stakeholder decision making

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Platform: GEE](https://img.shields.io/badge/Platform-Google%20Earth%20Engine-green.svg)](https://earthengine.google.com/)
[![Data: Planet](https://img.shields.io/badge/Data-NICFI%20Planet-orange.svg)](https://www.planet.com/nicfi/)

<img src="figures/study_area.png" alt="Study area: Sundarbans mangrove forest" width="650"/>

<sub>Study area: The Sundarbans mangrove forest and the BFD ranges are shown on the study area map. Mangrove and non-mangrove sample points that were used for this study are illustrated. The Sundarbans, one of the world’s largest continuous mangrove forest and a demanding test case (for our study) for evaluating spatial uncertainty in environmental maps. The region has experienced sustained canopy degradation driven by altered freshwater flows, land-use pressures, and salinity intrusion. In this context, identifying where model predictions are reliable (and where they are not) is important for stakeholders for effective monitoring and decision-making..</sub>

</div>

---

## This repository contains the code accompanying the paper:
**Islam et al. (2025)**  
*Accuracy is not certainty: using model agreement and human judgment to assess spatial uncertainty in high-resolution mangrove mapping*

---

## Overview

Environmental maps derived from remote sensing and machine learning are often interpreted as definitive representations of reality. However, high classification accuracy does not necessarily imply spatially reliable knowledge. Accuracy metrics are often taken for granted as sufficient justification for producing and using binary maps, even though they summarize performance globally and do not reveal where predictions are stable or where they are unreliable. This gap is rarely interrogated in the literature, yet it is critical for decision-making in spatial contexts.

This repository implements a workflow that reframes classification outputs as **probabilistic representations of epistemic stability**, using:

- multiple base learners  
- stacked generalization  
- spatially explicit measures of model agreement  

Rather than asking *“what is the predicted class?”*, this work asks:

> **Where are predictions stable, and where are they fundamentally uncertain?**

The approach demonstrates that **continuous ensemble probabilities encode structured gradients of certainty**, where:
- probabilities near 0 or 1 indicate strong model agreement and high interpretability  
- intermediate probabilities indicate disagreement among models and ambiguity in human judgment  

---

## Key contributions

- Introduces a **model-pluralistic framework** for spatial uncertainty assessment
- Uses **base learner disagreement (standard deviation)** as a proxy for epistemic uncertainty, that is, uncertainty due to limited knowledge, which appears as disagreement among multiple plausible models trained on the same data
- Demonstrates that **stacked probabilities capture structured uncertainty gradients of base learners**
- Compares model-derived uncertainty with **independent human interpretation**
- Moves beyond accuracy metrics toward **decision-relevant uncertainty mapping**

Although demonstrated for mangrove mapping, this framework is not domain-specific. It can be generalized to other mapping problems where uncertainty itself carries decision value. For example, in damage mapping, building-level damage probabilities can guide prioritization of field assessment and resource allocation; in disaster risk zoning, probability surfaces can represent gradients of perceived or modeled risk rather than fixed boundaries; in snow depth mapping, estimates can be paired with uncertainty bounds to inform hydrological forecasting and water resource planning. In each case, the objective shifts from producing a single definitive map to representing where predictions are stable and where knowledge remains uncertain.

---

## Does model uncertainty aligns with human interpretation?

![Interpreter vs Model](figures/fig_scatterplot.png)

*Fig1: Scatterplots of interpreter mean values versus model-predicted probabilities for four stacked generalization configurations (Stacking-LogReg-NPT, Stacking-LogReg-PT, Stacking-RF-NPT, Stacking-RF-PT).*

Stacked probability maps encode disagreement among base learners as probabilistic uncertainty. To evaluate whether these probabilities correspond to meaningful uncertainty in real-world interpretation, we compared model predictions against independent human interpretation scores.

Across all stacking configurations, model probabilities show a strong positive relationship with interpreter confidence. Pixels with high model agreement align closely with interpreter judgments near the extremes of absence or presence, while pixels with weaker agreement exhibit greater dispersion around the 1:1 relationship.

---

## Spatial variability in classification represented by stacked generalized map

![Spatial Probability Map](figures/fig_spatial.png)

*Fig2: a. Spatial distribution of mangrove probability derived from stacked generalization (Random Forest, no feature pass-through), with b. zoomed examples and c. comparison to Global Mangrove Watch (GMW) and d. MAXAR high-resolution imagery. The values indicates probability range from 0 (dark gray; very likely non-mangrove) to 1 (pale yellow; very likely mangrove).*

The final probability map represents a continuous surface of mangrove likelihood, where values range from 0 (very likely non-mangrove) to 1 (very likely mangrove). This configuration was selected due to its strong alignment with high-confidence human interpretation while remaining interpretable.

High-probability regions correspond primarily to closed-canopy mangrove areas, whereas lower and intermediate values are concentrated in more heterogeneous environments, particularly in western portions of the forest with open canopy structure.

Zoomed examples reveal that intermediate probabilities (≈0.3–0.7) are not randomly distributed, but occur systematically along geomorphological features such as river confluences, tidal creeks, and drainage channels. These areas frequently correspond to mixed or transitional vegetation conditions.

Compared to the Global Mangrove Watch (GMW) binary product, the continuous probability map better resolves non-mangrove areas and captures fine-scale landscape structure visible in high-resolution MAXAR imagery.

These patterns show that uncertainty is spatially organized and closely linked to environmental heterogeneity, and provide actionable information for targeted validation and monitoring.

---
## Repository structure
```
uncertaintypaper_codes/
│
├── README.md
├── LICENSE
│
├── GEE_codes/                                          # Google Earth Engine JavaScript scripts
│   ├── featureextract.js                               # Extract spectral features from Planet NICFI composites
│   ├── featureextract_CCDC.js                          # Extract CCDC temporal segmentation coefficients
│   ├── interfaceforvisualizinginterpretervalues.js     # Interactive GEE app for visualizing interpreter scores
│   ├── pointswheremodeldisagreed.js                    # Visualize locations of model disagreement
│   └── pointswheremodelsagreed.js                      # Visualize locations of model agreement
│
├── figures/                                            # Figures used in the README and paper
│   ├── study_area.png                                  # Study area map of the Sundarbans
│   ├── fig_scatterplot.png                             # Interpreter vs. model probability scatterplots
│   └── fig_spatial.png                                 # Spatial probability map from stacked generalization
│
└── python_codes/                                       # All Python-based analysis
    │
    ├── baselearnerSD_cleaned.ipynb                     # Compute per-pixel standard deviation across base learners
    ├── hero_scatter_cleaned.ipynb                      # Generate interpreter vs. model scatterplots (Fig. 1)
    ├── sd_compare clear.ipynb                          # Compare SD distributions across stacking configurations
    │
    └── stacked_generalization/                         # Core modeling pipeline
        │
        ├── conda_env.txt                              # Conda environment specification for reproducibility
        ├── .gitignore
        │
        ├── data/
        │   └── SamplePointsExport_nicfi_all.csv       # Reference sample points exported from GEE
        │
        ├── scripts/                                   # Numbered scripts run in sequence (01-08)
        │   ├── 01_compute_correlogram.R               # Compute spatial correlogram to determine CV block size
        │   ├── 02_generate_cv_blocks.R                # Generate spatial CV blocks (~25 km) using blockCV
        │   ├── 03_format_modeling_datasets.ipynb/.py   # Format raw data into modeling-ready train/test splits
        │   ├── 04_tune_baselearners.ipynb/.py          # Hyperparameter tuning with Optuna for each base learner
        │   ├── 05_run_repeated_cv.ipynb/.py            # Run repeated spatial cross-validation (100 iterations)
        │   ├── 06_format_cv_summary_table.ipynb/.py    # Summarize CV results into accuracy tables
        │   ├── 07_train_full_models.ipynb/.py          # Train final base learners and stacking meta-learners
        │   ├── 08_apply_models.ipynb/.py               # Apply trained models to generate probability surfaces
        │   │
        │   ├── helpers/                               # Supporting scripts for batch processing and GEE export
        │   │   ├── export_feature_layers.js           # GEE script to export feature layers as rasters
        │   │   ├── mosaic_planet_features.py          # Mosaic exported Planet feature tiles
        │   │   ├── mosaic_stacking_features.py        # Mosaic stacking feature tiles
        │   │   ├── run_all_modeling_scripts.py         # Batch runner for the full modeling pipeline
        │   │   ├── run_all_inference_scripts.py        # Batch runner for the full inference pipeline
        │   │   └── 03h_make_basic_boxplots.py          # Generate boxplots of feature distributions
        │   │
        │   └── utils/                                 # Shared utility modules
        │       ├── constants.py                       # Global constants (random seed, Optuna trial count, etc.)
        │       ├── helpers.py                         # Evaluation metrics, calibration, and scoring functions
        │       └── optimize_models.py                 # Optuna objective functions for each base learner
        │
        ├── intermediate_outputs/                      # Intermediate data products (auto-generated)
        │   ├── raw_tuning_dataset.csv                 # Raw tuning data before formatting
        │   ├── tuning_dataset.csv                     # Formatted tuning dataset
        │   ├── spatial_blocks_tuning.geojson           # Spatial block geometries for CV
        │   ├── raw_repeated_cv_datasets/              # 100 raw CV split datasets (cv_dataset_1..100.csv)
        │   └── formatted_repeated_cv_datasets/        # 100 formatted CV split datasets (cv_dataset_1..100.csv)
        │
        ├── models/                                    # Serialized model artifacts (.joblib)
        │   ├── scaler.joblib                          # Fitted feature scaler
        │   ├── baselearners/                          # Trained base learner and stacking meta-learner models
        │   │   ├── baselearner_{knn,logreg,rf,svc,xgb}_model.joblib
        │   │   └── stacking_{logreg,rf}_{np,npt}.joblib
        │   └── optuna_runs/                           # Optuna study objects for each base learner
        │       └── baselearner_{knn,logreg,rf,svc,xgb}_optuna.joblib
        │
        └── outputs/                                   # Final results and figures
            ├── best_baselearner_hyperparameters.csv    # Best hyperparameters selected by Optuna
            ├── model_cv_accuracy.csv                  # Raw CV accuracy results
            ├── formatted_cv_accuracy.csv              # Formatted accuracy summary table
            ├── formatted_cv_accuracy_calibration.csv  # Accuracy summary with calibration applied
            ├── formatted_cv_no_calibration.csv        # Accuracy summary without calibration
            ├── correlogram.RData                      # Saved correlogram object from R
            ├── correlogram_plot.png                   # Correlogram plot used to determine block size
            └── figure_folder/                         # Base learner performance metric plots
                └── baselearner_{Accuracy,AUC,F1-Score,Precision,Recall}.PNG
```

---

## Workflow

1. **Preprocessing**
   - Apply masking to remove clouds, haze, and tidal effects
   - Generate seasonal composites

2. **Feature Engineering**
   - Compute spectral bands and vegetation indices
   - Extract temporal features using CCDC (Continuous Change Detection and Classification; a temporal segmentation algorithm) coefficients

3. **Model Training**
   - Train multiple base learners:
     - Random Forest
     - XGBoost
     - Support Vector Classifier
     - KNN
     - Logistic Regression
   - Use spatial cross-validation (BlockCV, ~25 km blocks)

4. **Stacked Generalization**
   - Combine base learner predictions into continuous probability surfaces
   - Evaluate stacking with different meta-learners and configurations

5. **Uncertainty Quantification**
   - Compute **per-pixel standard deviation** across base learners
   - Identify regions of:
     - agreement (low SD)
     - disagreement (high SD)

6. **Analysis of Epistemic Stability**
   - Understanding base learner variability with stacked probabilities
   - Examine how uncertainty is distributed spatially

7. **Human Interpretation Comparison**
   - Evaluate whether model uncertainty aligns with human-perceived ambiguity
   - Use blinded interpreters assigning continuous confidence scores

---

## Methods highlights

- **Spatial cross-validation**  
  BlockCV with ~25 km spatial structure to account for autocorrelation

- **Model ensemble design**  
  Multiple base learners reflecting different inductive biases

- **Stacked generalization**  
  Integrates predictions while preserving disagreement signals

- **Probability calibration (evaluated)**  
  Platt scaling, isotonic regression, and beta calibration

- **Uncertainty representation**  
  Standard deviation across base learners used as a proxy for epistemic uncertainty

---

## Data

- NICFI PlanetScope basemaps (4.77 m resolution) multi-temporal image collection and derived spectral indices and CCDC features [(Planet NICFI Asia Basemaps available on GEE)](https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_asia#description)

#### Base learner inference maps
- `projects/ee-islamkm/assets/baselearner_knn_mngrv`
- `projects/ee-islamkm/assets/baselearner_logreg_mngrv`
- `projects/ee-islamkm/assets/baselearner_rf_mngrv`
- `projects/ee-islamkm/assets/baselearner_svc_mgrv`
- `projects/ee-islamkm/assets/baselearner_xgb_mgrv`

#### Stacked generalization inference maps
- `projects/ee-ashrafulcuetbd/assets/stacking_logreg_npt_prediction`
- `projects/ee-ashrafulcuetbd/assets/stacking_logreg_pt_prediction`
- `projects/ee-ashrafulcuetbd/assets/stacking_rf_npt_prediction`
- `projects/ee-ashrafulcuetbd/assets/stacking_rf_pt_prediction`

These assets store model-derived probability surfaces used for uncertainty analysis, comparison with interpreter scores, and generation of final probability maps.
(Data access may depend on external platforms such as Google Earth Engine.)

---

## Notes

- Spatial cross-validation is repeated multiple times for stability
- Hyperparameter tuning performed using Optuna
- Results emphasize consistency across model configurations rather than single-model performance

---

## Citation

If you use this code, please cite:
Islam, K. M. A., Kilbride, J. B., Murillo-Sandoval, P. J., & Kennedy, R. E. (2025).
Accuracy is not certainty: using model agreement and human judgment to assess spatial uncertainty in high-resolution mangrove mapping.
