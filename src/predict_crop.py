import numpy as np
import pickle
import os


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Load the trained model and label encoder
    model_path = os.path.join("..", "models", "crop_recommendation_model.pkl")
    le_path = os.path.join("..", "models", "label_encoder.pkl")

    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    with open(le_path, 'rb') as file:
        loaded_le = pickle.load(file)

    # Create a numpy array from the user inputs
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Get the crop name from the label encoder
    recommended_crop = loaded_le.inverse_transform(prediction)

    return recommended_crop[0]


# Example Prediction
print("\nRecommended Crop:", predict_crop(90, 42, 43, 20.5, 82, 6.5, 202))

#
# import requests
# import json
#
#
# def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
#     url = "http://127.0.0.1:5000/predict"
#     data = {
#         "N": N,
#         "P": P,
#         "K": K,
#         "temperature": temperature,
#         "humidity": humidity,
#         "ph": ph,
#         "rainfall": rainfall
#     }
#     headers = {'Content-Type': 'application/json'}
#     response = requests.post(url, data=json.dumps(data), headers=headers)
#
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {"error": "Failed to get response from server"}
#
#
# if __name__ == "__main__":
#     # Example input values
#     N = 50
#     P = 30
#     K = 40
#     temperature = 25.5
#     humidity = 70
#     ph = 6.5
#     rainfall = 100
#
#     result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
#     print("Prediction Result:", result)
#
