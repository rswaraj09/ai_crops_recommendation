from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("D:/H2C/crop-recommendation-system/data/random_merged_soil_dataset.csv")


# API to get NPK values for a district
@app.route('/get_soil_data', methods=['GET'])
def get_soil_data():
    district = request.args.get('district')

    # Filter dataset by district
    soil_data = df[df['District'].str.lower() == district.lower()]

    if soil_data.empty:
        return jsonify({"error": "District not found"}), 404

    # Get first matching row
    soil_info = soil_data.iloc[0][['N', 'P', 'K', 'ph', 'EC']].to_dict()
    return jsonify(soil_info)


# API to predict crop based on soil parameters
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.json

    # Filter dataset for similar soil conditions
    match = df[(df['N'] == data['N']) & (df['P'] == data['P']) & (df['K'] == data['K'])]

    if match.empty:
        return jsonify({"error": "No matching crop found"}), 404

    # Return the most common crop label
    predicted_crop = match['label'].mode()[0]
    return jsonify({"predicted_crop": predicted_crop})


if __name__ == '__main__':
    app.run(debug=True)
