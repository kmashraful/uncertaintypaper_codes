"""
Script Name: 04b_get_baselearner_predictions.py
Author: John Kilbride
Date: 2025-03-16
Description: 
    
    This script runs repeated cross-validation using the tuning dataset to identify
    the optimal hyperparameters for the Random Forest, SVM, KNN, ElasticNet, and
    XGBoosting classifiers. 
    
    
"""

import warnings
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from utils.helpers import train_and_calibrate_model
from utils.constants import FEATURES

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.core")


if __name__ == "__main__":
   
    # Define the folder where intermediate outputs are be stored
    intermediate_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/intermediate_outputs"

    # Define the folder wher outputs are stored
    output_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/outputs"

    # Define the folder that contain repeated CV datasets
    repeated_cv_folder = f"{intermediate_folder}/formatted_repeated_cv_datasets"
    
    # Define the output infold and out-of-fold (oof) folders
    infold_cv_folder = f"{intermediate_folder}/infold_repeated_cv_datasets"
    oof_cv_folder = f"{intermediate_folder}/oof_repeated_cv_datasets"
   
    # Read on the optimal hyperparameters
    best_hyperparams_df = pd.read_csv(f"{output_folder}/best_baselearner_hyperparameters.csv")

    ###########################################################################
    ### STEP 1. UNPACK THE OPTIMAL BASELEARNER HYPERPARAMETERS
    ###########################################################################

    # Parse each model's parameters from the DataFrame into dictionaries
    best_rf_params = literal_eval(
        best_hyperparams_df.loc[best_hyperparams_df["Model"] == "Random Forest", "Params"].iloc[0]
        )
    best_svc_params = literal_eval(
        best_hyperparams_df.loc[best_hyperparams_df["Model"] == "SVC", "Params"].iloc[0]
        )
    best_knn_params = literal_eval(
        best_hyperparams_df.loc[best_hyperparams_df["Model"] == "KNN", "Params"].iloc[0]
        )
    best_logreg_params = literal_eval(
        best_hyperparams_df.loc[best_hyperparams_df["Model"] == "Logistic Regression", "Params"].iloc[0]
        )
    best_xgb_params = literal_eval(
        best_hyperparams_df.loc[best_hyperparams_df["Model"] == "XGBoost", "Params"].iloc[0]
        )

    ###########################################################################
    ### STEP 2. DEFINE THE BASELEARNERS TO ASSESS
    ###########################################################################    
    
    rf_model = RandomForestClassifier(**best_rf_params)
    svc_model = SVC(**best_svc_params, probability=True)
    knn_model = KNeighborsClassifier(**best_knn_params)
    logreg_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=10000,
        **best_logreg_params
        )
    xgb_model = XGBClassifier(
        **best_xgb_params, 
        objective = "binary:logistic", 
        eval_metric = "logloss"
        )

    ###########################################################################
    ### STEP 3. DEFINE THE STACKED GENERALIZATION MODELS TO TEST
    ###########################################################################

    # Define base learners including logistic regression
    base_learners = [
        ('rf', rf_model),
        ('svc', svc_model),
        ('knn', knn_model),
        ('logreg', logreg_model),
        ('xgb', xgb_model)
        ]
    
    # Define a final logistic regression as the final estimator
    stacking_logreg_model = LogisticRegression(
        penalty = None,
        solver = 'newton-cg', 
        max_iter = 10000
        )    
    
    # Define Random Forest classifier
    stacking_rf_model = RandomForestClassifier(
        n_estimators = 500,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_features = 'sqrt',
        bootstrap = True
        )
    
    # Create a stacking model w/ no passthrough features and 
    # w/ logistic regression stacking model
    stacking_model_logreg_npt = StackingClassifier(
        estimators = base_learners,
        final_estimator = stacking_logreg_model,
        cv = 5,  
        stack_method = "predict_proba",
        passthrough = False,
        n_jobs = -1
        )
    
    # Create a stacking model w/ passthrough features and 
    # w/ logistic regression stacking model
    stacking_model_logreg_pt = StackingClassifier(
        estimators = base_learners,
        final_estimator = stacking_logreg_model,
        cv = 5,  
        stack_method = "predict_proba",
        passthrough = True,
        n_jobs = -1
        )
    
    # Create a stacking model w/ no passthrough features and 
    # w/ Random Forests stacking model
    stacking_model_rf_npt = StackingClassifier(
        estimators = base_learners,
        final_estimator = stacking_rf_model,
        cv = 5,  
        stack_method = "predict_proba",
        passthrough = False,
        n_jobs = -1
        )

    # Create a stacking model w/ passthrough features and 
    # w/ Random Forests stacking model
    stacking_model_rf_pt = StackingClassifier(
        estimators = base_learners,
        final_estimator = stacking_rf_model,
        cv = 5,  
        stack_method = "predict_proba",
        passthrough = True,
        n_jobs = -1
        )
    
    ###########################################################################
    ### STEP 3. RUN THE REPEATED CROSS VALIDATION PROCEDURE
    ###########################################################################
    
    # Initialize an empty list to store accuracy statistics
    baselearner_accuracy = []
    
    # Iterate over the different datasets
    for cv_iter_i in tqdm(range(1,101)):
                
        # Load in the training dataset
        training_df = pd.read_csv(f"{repeated_cv_folder}/cv_dataset_{cv_iter_i}.csv")
    
        # Loop over each fold
        for fold in training_df["cv_fold_id"].unique():
            
            # Get the training/testing dataset
            train_data = training_df[training_df["cv_fold_id"] != fold]
            test_data = training_df[training_df["cv_fold_id"] == fold]
            
            # Random forests
            rf_metrics = train_and_calibrate_model(
                model = rf_model,
                model_name = "RandomForest",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )
            baselearner_accuracy += rf_metrics
 
            # Support vector classifier
            svc_metrics = train_and_calibrate_model(
                model = svc_model,
                model_name = "SVC",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )
            baselearner_accuracy += svc_metrics

            # K-Nearest Neighbors
            knn_metrics = train_and_calibrate_model(
                model = knn_model,
                model_name = "KNN",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )  
            baselearner_accuracy += knn_metrics
            
            # Logistic Regression w/ ElasticNet penalty
            knn_metrics = train_and_calibrate_model(
                model = logreg_model,
                model_name = "LogisticRegression",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )  
            baselearner_accuracy += knn_metrics
            
            # XGBoost
            xgb_metrics = train_and_calibrate_model(
                model = xgb_model,
                model_name = "XGBoost",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )  
            baselearner_accuracy += xgb_metrics
            
            # Stacking w/ Logistic Regression & no passthrough
            stacking_logreg_npt_metrics = train_and_calibrate_model(
                model = stacking_model_logreg_npt,
                model_name = "Stacking-LogReg-NPT",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )  
            baselearner_accuracy += stacking_logreg_npt_metrics
            
            # Stacking w/ Logistic Regression & passthrough
            stacking_logreg_pt_metrics = train_and_calibrate_model(
                model = stacking_model_logreg_pt,
                model_name = "Stacking-LogReg-PT",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )  
            baselearner_accuracy += stacking_logreg_pt_metrics
            
            # Stacking w/ Random Forests & no passthrough
            stacking_rf_npt_metrics = train_and_calibrate_model(
                model = stacking_model_rf_npt,
                model_name = "Stacking-RF-NPT",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )  
            baselearner_accuracy += stacking_rf_npt_metrics
            
            # Stacking w/ Random Forests & passthrough
            stacking_rf_pt_metrics = train_and_calibrate_model(
                model = stacking_model_rf_pt,
                model_name = "Stacking-RF-PT",
                train_data = train_data,
                test_data = test_data,
                features = FEATURES,
                response = "landcover",
                fold = fold,
                cv_id = cv_iter_i 
                )  
            baselearner_accuracy += stacking_rf_pt_metrics
            
    # Combine and save the baselearner accuracy statistics
    metrics_df = pd.DataFrame(baselearner_accuracy)
    metrics_df =  metrics_df[[
        'Classifier_Type', 'CV_ID', 'Fold', 'Accuracy', 'F1_Score', 'AUC', 
        'Precision', 'Recall', 'logloss'
        ]]
    metrics_df = metrics_df.sort_values(by = ['Classifier_Type', 'CV_ID', 'Fold'])
    metrics_df.to_csv(f"{output_folder}/model_cv_accuracy.csv", index=False)
