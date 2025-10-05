# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pickle
# from sklearn.preprocessing import LabelEncoder
#
# def train_model():
#     # Load the dataset
#     data = pd.read_csv("D:/H2C/crop-recommendation-system/data/crop_recommendation.csv")
#
#     # Encode categorical data (Label Encoding for Crops)
#     le = LabelEncoder()
#     data['label'] = le.fit_transform(data['Crop'])  # Ensure column name matches CSV
#
#     # Map soil types to numeric values
#     soil_mapping = {
#         "sandy": 0,
#         "clay": 1,
#         "silt": 2,
#         "loamy": 3
#     }
#     data['soil_type'] = data['Soil Type'].map(soil_mapping)
#
#     # Select features and target variable
#     X = data[['soil_type', 'temperature', 'humidity', 'rainfall']]
#     y = data['label']
#
#     # Split dataset
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Train the Random Forest model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#
#     # Evaluate model
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Model Accuracy: {accuracy * 100:.2f}%")
#
#     # Save trained model and label encoder
#     model_dir = "../models/"
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#
#     with open(model_dir + "crop_recommendation_model.pkl", "wb") as file:
#         pickle.dump(model, file)
#
#     with open(model_dir + "label_encoder.pkl", "wb") as file:
#         pickle.dump(le, file)
#
#     print("Model and Label Encoder saved successfully!")
#
# # Train model when script is run
# if __name__ == "__main__":
#     train_model()


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("D:/H2C/crop-recommendation-system/data/random_merged_soil_dataset.csv")

# Select relevant features and target
features = ["N", "P", "K", "ph", "Zn", "Fe", "Cu", "Mn", "B", "S"]
X = data[features]
y = data["label"]  # Ensure this column exists

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "crop_prediction_model.pkl")

print("Model saved as crop_prediction_model.pkl")
