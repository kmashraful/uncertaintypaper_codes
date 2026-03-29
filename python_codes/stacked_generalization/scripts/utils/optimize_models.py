import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from .constants import RANDOM_SEED


def optimize_rf(trial, training_df, target_col, features, cv_fold_col):
    """
    Optimize hyperparameters for RandomForestClassifier using Optuna.

    Parameters:
        trial (optuna.Trial): A single optimization trial.
        training_df (pd.DataFrame): Training dataset with scaled features.
        target_col (str): Name of the target column.
        features (list): List of feature column names.
        cv_fold_col (str): Name of the cross-validation fold column.

    Returns:
        float: Mean F1 score across cross-validation folds.
    """
    # Suggest hyperparameters for RandomForestClassifier
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_int("max_depth", 3, 1000)
    max_features = trial.suggest_float("max_features", 0.25, 1, log=False)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 50, log=False)

    # Initialize classifier with fixed max_depth and suggested max_features
    clf = RandomForestClassifier(
        n_estimators = 500,
        criterion = criterion,
        max_depth = max_depth,
        max_features = max_features,
        min_samples_split = min_samples_split,
        random_state = RANDOM_SEED,
        n_jobs = -1
        )

    scores = []

    # Perform cross-validation using cv_fold_col
    for fold in training_df[cv_fold_col].unique():
        train_data = training_df[training_df[cv_fold_col] != fold]
        valid_data = training_df[training_df[cv_fold_col] == fold]

        X_train = train_data[features].values
        y_train = train_data[target_col].values
        X_valid = valid_data[features].values
        y_valid = valid_data[target_col].values

        # Fit and evaluate
        clf.fit(X_train, y_train)
        preds = clf.predict(X_valid)
        scores.append(f1_score(y_valid, preds))

    # Return mean F1 score across folds
    return np.mean(scores)


def optimize_svc(trial, training_df, target_col, features, cv_fold_col):
    """
    Optimize hyperparameters for Support Vector Classifier (SVC) using Optuna.

    Parameters:
        trial (optuna.Trial): A single optimization trial.
        training_df (pd.DataFrame): Training dataset with scaled features.
        target_col (str): Name of the target column.
        features (list): List of feature column names.
        cv_fold_col (str): Name of the cross-validation fold column.

    Returns:
        float: Mean F1 score across cross-validation folds.
    """
    # Suggest hyperparameters for SVC
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
    
    # Set the degree if polynomial kernel is used
    if kernel == "poly":
        degree = trial.suggest_int("degree", 1, 6)
    else:
        degree = 3
        
    # Set coef0 if polynomial or sigmoidal kernel is used
    if kernel == "poly":
        coef0 = trial.suggest_float("coef0", 1e-3, 10, log=True)
    else:
        coef0 = 0
        

    # Initialize classifier with suggested hyperparameters
    clf = SVC(
        C = C, 
        gamma = gamma,
        kernel = kernel, 
        degree = degree,
        coef0 = coef0,
        probability = True, 
        random_state = RANDOM_SEED
        )

    scores = []

    # Perform cross-validation using cv_fold_col
    for fold in training_df[cv_fold_col].unique():
        train_data = training_df[training_df[cv_fold_col] != fold]
        valid_data = training_df[training_df[cv_fold_col] == fold]

        X_train = train_data[features].values
        y_train = train_data[target_col].values
        X_valid = valid_data[features].values
        y_valid = valid_data[target_col].values

        # Fit and evaluate
        clf.fit(X_train, y_train)
        preds = clf.predict(X_valid)
        scores.append(f1_score(y_valid, preds))

    # Return mean F1 score across folds
    return np.mean(scores)


def optimize_knn(trial, training_df, target_col, features, cv_fold_col):
    """
    Optimize hyperparameters for KNeighborsClassifier using Optuna.

    Parameters:
        trial (optuna.Trial): A single optimization trial.
        training_df (pd.DataFrame): Training dataset with scaled features.
        target_col (str): Name of the target column.
        features (list): List of feature column names.
        cv_fold_col (str): Name of the cross-validation fold column.

    Returns:
        float: Mean F1 score across cross-validation folds.
    """
    # Suggest hyperparameters for KNeighborsClassifier
    n_neighbors = trial.suggest_int("n_neighbors", 3, 29, step=2)
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])

    # Initialize classifier with suggested hyperparameters
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    scores = []

    # Perform cross-validation using cv_fold_col
    for fold in training_df[cv_fold_col].unique():
        train_data = training_df[training_df[cv_fold_col] != fold]
        valid_data = training_df[training_df[cv_fold_col] == fold]

        X_train = train_data[features].values
        y_train = train_data[target_col].values
        X_valid = valid_data[features].values
        y_valid = valid_data[target_col].values

        # Fit and evaluate
        clf.fit(X_train, y_train)
        preds = clf.predict(X_valid)
        scores.append(f1_score(y_valid, preds))

    # Return mean F1 score across folds
    return np.mean(scores)


def optimize_logreg_elasticnet(trial, training_df, target_col, features, cv_fold_col):
    """
    Optimize hyperparameters for Logistic Regression with elasticnet penalty using Optuna.

    Parameters:
        trial (optuna.Trial): A single optimization trial.
        training_df (pd.DataFrame): Training dataset with scaled features.
        target_col (str): Name of the target column.
        features (list): List of feature column names.
        cv_fold_col (str): Name of the cross-validation fold column.

    Returns:
        float: Mean F1 score across cross-validation folds.
    """
    # Suggest hyperparameters for elasticnet penalty
    C = trial.suggest_float("C", 1e-5, 1.0, log = True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    # Initialize classifier with suggested hyperparameters
    clf = LogisticRegression(
        penalty = "elasticnet",
        solver = "saga",
        C = C,
        l1_ratio = l1_ratio,
        max_iter = 10000,
        random_state = RANDOM_SEED,
        )

    scores = []

    # Perform cross-validation using cv_fold_col
    for fold in training_df[cv_fold_col].unique():
        train_data = training_df[training_df[cv_fold_col] != fold]
        valid_data = training_df[training_df[cv_fold_col] == fold]

        X_train = train_data[features].values
        y_train = train_data[target_col].values
        X_valid = valid_data[features].values
        y_valid = valid_data[target_col].values

        # Fit and evaluate
        clf.fit(X_train, y_train)
        preds = clf.predict(X_valid)
        scores.append(f1_score(y_valid, preds))

    # Return mean F1 score across folds
    return np.mean(scores)


def optimize_xgboost(trial, training_df, target_col, features, cv_fold_col):
    """
    Optimize hyperparameters for XGBoost using Optuna with the scikit-learn API.

    Parameters:
        trial (optuna.Trial): A single optimization trial.
        training_df (pd.DataFrame): Training dataset with scaled features.
        target_col (str): Name of the target column.
        features (list): List of feature column names.
        cv_fold_col (str): Name of the cross-validation fold column.

    Returns:
        float: Mean F1 score across cross-validation folds.
    """
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "gamma": trial.suggest_float("gamma", 1e-5, 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "min_child_weight": trial.suggest_float("min_child_weight", 0, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.05, 1.0),
        "reg_lambda": trial.suggest_float("lambda", 1e-5, 1, log=True),
        "reg_alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
        "random_state": RANDOM_SEED,
        "n_estimators": trial.suggest_int("n_estimators", 15, 1500),
        "n_jobs": -1
    }

    scores = []
    for fold in training_df[cv_fold_col].unique():
        train_data = training_df[training_df[cv_fold_col] != fold]
        valid_data = training_df[training_df[cv_fold_col] == fold]

        X_train = train_data[features].values
        y_train = train_data[target_col].values
        X_valid = valid_data[features].values
        y_valid = valid_data[target_col].values

        model = XGBClassifier(**param)
        model.fit(X_train, y_train)

        preds = model.predict(X_valid)
        scores.append(f1_score(y_valid, preds))

    return np.mean(scores)
