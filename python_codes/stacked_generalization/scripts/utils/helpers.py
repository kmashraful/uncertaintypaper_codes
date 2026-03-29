import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from betacal import BetaCalibration
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from typing import Union, Any, List, Dict


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred_probs: Union[np.ndarray, pd.Series], 
    threshold: float = 0.5
    ) -> Dict[str, float]:
    """
    Calculate classification metrics given the ground truth and predicted probabilities.

    Args:
        y_true (Union[np.ndarray, pd.Series]): Ground truth labels.
        y_pred_probs (Union[np.ndarray, pd.Series]): Predicted probabilities.
        threshold (float, optional): Probability threshold for classifying as positive. Default is 0.5.

    Returns:
        Dict[str, float]: A dictionary containing Accuracy, F1_Score, AUC, Precision, and Recall.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)  # Convert probabilities to binary predictions

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred_probs),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }
    
    return metrics


def train_and_calibrate_model(
    model: Any,
    model_name: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: List[str],
    response: str,
    fold: int,
    cv_id: int
    ) -> List[Dict[str, Any]]:
    """
    Train a classification model and apply different calibration techniques to improve probability estimates.
    
    This function trains the provided model on training data and evaluates its performance on test data.
    It then applies various probability calibration methods including Platt scaling, Beta calibration (ab & abm),
    and Isotonic regression to refine the predicted probabilities. The performance of each calibration method
    is evaluated using classification metrics.
    
    Args:
        model (Any): A scikit-learn-compatible classifier supporting `fit` and `predict_proba`.
        model_name (str): The name of the model for tracking in results.
        train_data (pd.DataFrame): The training dataset containing features and response.
        test_data (pd.DataFrame): The test dataset containing features and response.
        features (List[str]): List of feature column names to use for training.
        response (str): The response column name (target variable).
        fold (int): Fold number for cross-validation tracking.
        cv_id (int): Cross-validation identifier.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing classification metrics for different calibration methods.
    """    
    # Prepare data
    X_train = train_data[features].values
    y_train = train_data[response].values
    X_test = test_data[features].values
    y_test = test_data[response].values
    
    # Fit the model using scikit-learn or XGBoost
    model.fit(X_train, y_train)
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # List to hold metrics for each calibration type
    stacking_accuracy = []
    
    # Uncalibrated
    uncal_metrics = calculate_metrics(y_test, test_probs)
    uncal_metrics.update({
        "Fold": fold,
        "CV_ID": cv_id, 
        "Classifier_Type": f"{model_name}-Uncalibrated",
        "logloss": log_loss(y_test, test_probs)
    })
    stacking_accuracy.append(uncal_metrics)
    
    # Platt scaling
    sigmoid_model = LogisticRegression(penalty=None)
    sigmoid_model.fit(train_probs.reshape(-1, 1), y_train)
    platt_probs = sigmoid_model.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    platt_metrics = calculate_metrics(y_test, platt_probs)
    platt_metrics.update({
        "Fold": fold,
        "CV_ID": cv_id,
        "Classifier_Type": f"{model_name}-Platt",
        "logloss": log_loss(y_test, platt_probs)
    })
    stacking_accuracy.append(platt_metrics)
    
    # Beta Calibration (ab)
    beta_calibration_ab = BetaCalibration(parameters="ab")
    beta_calibration_ab.fit(train_probs.reshape(-1, 1), y_train)
    beta_ab_probs = beta_calibration_ab.predict(test_probs.reshape(-1, 1))
    beta_ab_metrics = calculate_metrics(y_test, beta_ab_probs)
    beta_ab_metrics.update({
        "Fold": fold,
        "CV_ID": cv_id,
        "Classifier_Type": f"{model_name}-Beta-ab",
        "logloss": log_loss(y_test, beta_ab_probs)
    })
    stacking_accuracy.append(beta_ab_metrics)
    
    # Beta Calibration (abm)
    beta_calibration_abm = BetaCalibration(parameters="abm")
    beta_calibration_abm.fit(train_probs.reshape(-1, 1), y_train)
    beta_abm_probs = beta_calibration_abm.predict(test_probs.reshape(-1, 1))
    beta_abm_metrics = calculate_metrics(y_test, beta_abm_probs)
    beta_abm_metrics.update({
        "Fold": fold,
        "CV_ID": cv_id,
        "Classifier_Type": f"{model_name}-Beta-abm",
        "logloss": log_loss(y_test, beta_abm_probs)
    })
    stacking_accuracy.append(beta_abm_metrics)
    
    # Isotonic Regression
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(train_probs, y_train)
    isotonic_probs = isotonic_model.predict(test_probs)
    iso_metrics = calculate_metrics(y_test, isotonic_probs)
    iso_metrics.update({
        "Fold": fold,
        "CV_ID": cv_id,
        "Classifier_Type": f"{model_name}-Isotonic",
        "logloss": log_loss(y_test, isotonic_probs)
    })
    stacking_accuracy.append(iso_metrics)
    
    return stacking_accuracy


