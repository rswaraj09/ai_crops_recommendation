import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:/H2C/main/H2C/crop-recommendation-system/data/random_merged_soil_dataset.csv")

# Features and labels
X = df[['N', 'P', 'K', 'ph', 'EC']]  # Adjust based on available columns
y = df['label']  # Column that contains crop names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("C:/H2C/main/H2C/crop-recommendation-system/src/models/crop_prediction_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")