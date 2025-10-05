"""
Train the Image Classification and Soil Parameter Prediction Models

This script handles the training of all machine learning models for the crop recommendation system:
1. Image classification model for soil type recognition
2. Soil parameter prediction model to extract N, P, K, pH values from soil images
3. Crop recommendation model based on soil and weather parameters

The script includes data preprocessing, feature extraction, model training, and evaluation.
"""

import os
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling required dependencies...")
    dependencies = [
        "pandas", "numpy", "scikit-learn", "opencv-python", "matplotlib", 
        "joblib", "scikit-image", "openpyxl"
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"Successfully installed {dep}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}")

def load_crop_recommendation_data():
    """Load and prepare crop recommendation dataset"""
    data_path = os.path.join("data", "crop_recommendation.csv")
    if not os.path.exists(data_path):
        print(f"Error: Crop recommendation dataset not found at {data_path}")
        return None, None, None
    
    print(f"Loading crop recommendation data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Encode the crop labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['label'])
        X = df.drop('label', axis=1)
        
        # Save crop labels for reference
        labels_df = pd.DataFrame({
            'index': range(len(label_encoder.classes_)),
            'crop': label_encoder.classes_
        })
        labels_df.to_csv(os.path.join("src", "models", "crop_labels.csv"), index=False)
        print(f"Saved crop labels mapping to src/models/crop_labels.csv")
        
        return X, y, label_encoder
    except Exception as e:
        print(f"Error loading crop recommendation data: {e}")
        return None, None, None

def load_soil_params_data():
    """Load soil parameters dataset"""
    data_path = os.path.join("data", "random_merged_soil_dataset.csv")
    if not os.path.exists(data_path):
        print(f"Error: Soil parameters dataset not found at {data_path}")
        return None
    
    print(f"Loading soil parameters data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Soil dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading soil parameters data: {e}")
        return None

def train_crop_recommendation_model(X, y, label_encoder):
    """Train the crop recommendation model"""
    print("\nTraining crop recommendation model...")
    
    # Create directory for models
    os.makedirs(os.path.join("src", "models"), exist_ok=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create a pipeline with preprocessing and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate the model on test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate feature importance
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nFeature importance:")
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save the model
    model_path = os.path.join("src", "models", "crop_recommendation_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Crop recommendation model saved to {model_path}")
    
    # Save the label encoder
    encoder_path = os.path.join("src", "models", "label_encoder.pkl")
    joblib.dump(label_encoder, encoder_path)
    print(f"Label encoder saved to {encoder_path}")
    
    return best_model

def train_soil_image_models():
    """Train the soil image analysis models"""
    print("\nTraining soil image analysis models...")
    
    # Run the dedicated script for training image models
    result = subprocess.run("python src/train_image_model.py", shell=True)
    
    if result.returncode == 0:
        print("Soil image analysis models training completed successfully!")
    else:
        print("Soil image analysis models training encountered some issues.")
        print("The app will use fallback predictions if the models aren't available.")

def main():
    """Main execution function"""
    print("=" * 80)
    print("Crop Recommendation System - Model Training")
    print("=" * 80)
    
    # Install required dependencies
    install_dependencies()
    
    # Create directories
    os.makedirs("src/models", exist_ok=True)
    
    # Load and prepare crop recommendation data
    X, y, label_encoder = load_crop_recommendation_data()
    if X is not None and y is not None and label_encoder is not None:
        # Train crop recommendation model
        train_crop_recommendation_model(X, y, label_encoder)
    
    # Train soil image analysis models
    train_soil_image_models()
    
    print("\nModel training completed!")
    print("You can now run the application with 'python run_app.py'")

if __name__ == "__main__":
    main() 