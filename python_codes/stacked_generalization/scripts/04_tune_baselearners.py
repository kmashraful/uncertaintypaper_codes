"""
Script Name: 04_tune_baselearners.py
Author: John Kilbride
Date: 2025-03-16
Description: 
    
    This script runs repeated cross-validation using the tuning dataset to identify
    the optimal hyperparameters for the Random Forest, SVM, KNN, ElasticNet, and
    XGBoosting classifiers. 
    
    
"""
import joblib
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from utils.optimize_models import optimize_rf
from utils.optimize_models import optimize_svc
from utils.optimize_models import optimize_knn
from utils.optimize_models import optimize_logreg_elasticnet
from utils.optimize_models import optimize_xgboost

from utils.constants import N_TRIALS, FEATURES


if __name__ == "__main__":
    
    # Define the folder where intermediate outputs are be stored
    intermediate_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/intermediate_outputs"

    # Define the folder wher outputs are stored
    output_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/outputs"

    # Define the folder where optuna models are saved
    optuna_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/models/optuna_runs"
    
    # Load in the training dataset
    training_df = pd.read_csv(f"{intermediate_folder}/tuning_dataset.csv")
   
    ###########################################################################
    ### STEP 1. TUNE THE BASE LEARNERS 
    ###########################################################################

    # Load in the Tree Parzen Estimator sampler
    sampler = TPESampler()

    # RandomForest optimization and training
    rf_study = optuna.create_study(direction="maximize", sampler=sampler)
    rf_study.optimize(
        lambda trial: optimize_rf(trial, training_df, "landcover", FEATURES, "cv_fold_id"),
        n_trials = N_TRIALS
        )
    best_rf_params = rf_study.best_params
    joblib.dump(rf_study, f"{optuna_folder}/baselearner_rf_optuna.joblib")

    # SVC optimization and training
    svc_study = optuna.create_study(direction="maximize", sampler=sampler)
    svc_study.optimize(
        lambda trial: optimize_svc(trial, training_df, "landcover", FEATURES, "cv_fold_id"),
        n_trials = N_TRIALS
        )
    best_svc_params = svc_study.best_params
    joblib.dump(svc_study, f"{optuna_folder}/baselearner_svc_optuna.joblib")

    # KNN optimization and training
    knn_study = optuna.create_study(direction="maximize", sampler=sampler)
    knn_study.optimize(
        lambda trial: optimize_knn(trial, training_df, "landcover", FEATURES, "cv_fold_id"),
        n_trials = N_TRIALS
        ) 
    best_knn_params = knn_study.best_params
    joblib.dump(knn_study, f"{optuna_folder}/baselearner_knn_optuna.joblib")
    
    # Logistic Regression optimization and training
    logreg_study = optuna.create_study(direction="maximize", sampler=sampler)
    logreg_study.optimize(
        lambda trial: optimize_logreg_elasticnet(trial, training_df, "landcover", FEATURES, "cv_fold_id"),
        n_trials = N_TRIALS
        )
    best_logreg_params = logreg_study.best_params
    joblib.dump(logreg_study, f"{optuna_folder}/baselearner_logreg_optuna.joblib")
    
    # XGBoost optimization and training
    xgb_study = optuna.create_study(direction="maximize", sampler=sampler)
    xgb_study.optimize(
        lambda trial: optimize_xgboost(trial, training_df, "landcover", FEATURES, "cv_fold_id"),
        n_trials = N_TRIALS
        )
    best_xgb_params = xgb_study.best_params
    joblib.dump(xgb_study, f"{optuna_folder}/baselearner_xgb_optuna.joblib")
    
    # Combine best hyperparameters for each model into a DataFrame
    results = [
        {"Model": "Random Forest", "Params": best_rf_params},
        {"Model": "SVC", "Params": best_svc_params},
        {"Model": "KNN", "Params": best_knn_params},
        {"Model": "Logistic Regression", "Params": best_logreg_params},
        {"Model": "XGBoost", "Params": best_xgb_params},
        ]    
    best_hyperparams_df = pd.DataFrame(results)
    best_hyperparams_df.to_csv(f"{output_folder}/best_baselearner_hyperparameters.csv", index=False)
