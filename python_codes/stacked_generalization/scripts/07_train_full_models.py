import joblib
import pandas as pd
from ast import literal_eval

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from utils.constants import RANDOM_SEED, FEATURES

if __name__ == "__main__":
   
    # Define the folder where intermediate outputs are be stored
    intermediate_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/intermediate_outputs"

    # Define the folder wher outputs are stored
    output_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/outputs"

    # Load in the training dataset (using the tuning dataset is fine as we use the whole thing)
    training_df = pd.read_csv(f"{intermediate_folder}/tuning_dataset.csv")
   
    # Read on the optimal hyperparameters
    best_hyperparams_df = pd.read_csv(f"{output_folder}/best_baselearner_hyperparameters.csv")
    
    # Define the folder where models
    models_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/models/baselearners"

    ###########################################################################
    ### STEP 1. UNPACK THE OPTIMAL HYPERPARAMETERS
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
    ### STEP 3. COMPUTE THE OUT-OF-FOLD META-FEATURES W/ OPTIMAL HYPERPARAMETERS
    ###########################################################################

    # Get the training dataset
    X_train = training_df[FEATURES].values
    y_train = training_df["landcover"].values
    
    #################################################
    # STEP 4. TRAIN THE MODELS
    #################################################
    
    # Random Forest
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, f"{models_folder}/baselearner_rf_model.joblib")

    # SVC
    svc_model.fit(X_train, y_train)
    joblib.dump(svc_model, f"{models_folder}/baselearner_svc_model.joblib")

    # KNN
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, f"{models_folder}/baselearner_knn_model.joblib")

    # Logistic Regression
    logreg_model.fit(X_train, y_train)
    joblib.dump(logreg_model, f"{models_folder}/baselearner_logreg_model.joblib")

    # XGBoost
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, f"{models_folder}/baselearner_xgb_model.joblib")
    
    # Stacking model w/ no passthrough and logistic regression stacking
    stacking_model_logreg_npt.fit(X_train, y_train)
    joblib.dump(stacking_model_logreg_npt, f"{models_folder}/stacking_logreg_npt.joblib")

    # Stacking model w/ passthrough and logistic regression stacking
    stacking_model_logreg_pt.fit(X_train, y_train)
    joblib.dump(stacking_model_logreg_pt, f"{models_folder}/stacking_logreg_np.joblib")

    # Stacking model w/ no passthrough and random forest stacking
    stacking_model_rf_npt.fit(X_train, y_train)
    joblib.dump(stacking_model_rf_npt, f"{models_folder}/stacking_rf_npt.joblib")

    # Stacking model w/ passthrough and random forest stacking
    stacking_model_rf_pt.fit(X_train, y_train)
    joblib.dump(stacking_model_rf_pt, f"{models_folder}/stacking_rf_np.joblib")
    