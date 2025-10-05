import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy metrics for the given true and predicted labels.
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate recall
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Return all metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model on the test dataset and calculate accuracy metrics.
    """
    # Predict the labels for the test dataset
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_accuracy(y_test, y_pred)
    
    # Return the metrics
    return metrics

def log_metrics(metrics):
    """
    Log the calculated metrics to a file for later analysis.
    """
    with open("accuracy_metrics.log", "a") as log_file:
        log_file.write("Accuracy Metrics:\n")
        for key, value in metrics.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")
