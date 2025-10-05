import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate dummy soil dataset (Replace with real dataset)
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
soil_model = RandomForestClassifier()
soil_model.fit(X_train, y_train)

# Save the trained model properly
soil_model_path = "D:/H2C/crop-recommendation-system/src/models/soil_model.pkl"
with open(soil_model_path, "wb") as file:
    pickle.dump(soil_model, file)

print("Soil model saved successfully!")
