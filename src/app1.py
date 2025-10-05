# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import pickle
# import os
#
# app = Flask(__name__)
#
# # Load dataset
# csv_path = "C:/H2C/main/H2C/crop-recommendation-system/data/random_merged_soil_dataset.csv"
# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path)
# else:
#     print("Error: Soil dataset file not found!")
#
# # Load the trained AI model
# model_path = "C:/H2C/main/H2C/crop-recommendation-system/src/models/crop_prediction_model.pkl"
# if os.path.exists(model_path):
#     with open(model_path, "rb") as model_file:
#         model = pickle.load(model_file)
# else:
#     print("Error: AI model file not found!")
#
#
# # Route for serving index.html
# @app.route('get.html')
# def get():
#     return render_template("get.html")  # Ensure "index.html" is inside a "templates" folder
#
#
# # API route for getting soil data
# @app.route('/get_soil_data', methods=['GET'])
# def get_soil_data():
#     district = request.args.get('district')
#
#     if not district:
#         return jsonify({"error": "District parameter is required"}), 400
#
#     # Ensure district column exists
#     if 'District' not in df.columns:
#         return jsonify({"error": "District column not found in dataset"}), 500
#
#     # Filter dataset by district
#     soil_data = df[df['District'].str.lower() == district.lower()]
#
#     if soil_data.empty:
#         return jsonify({"error": "District not found"}), 404
#
#     soil_info = soil_data.iloc[0][['N', 'P', 'K', 'ph', 'EC']].to_dict()
#     return jsonify(soil_info)
#
#
# # API route for predicting the crop
# @app.route('/predict_crop', methods=['GET'])
# def predict_crop():
#     try:
#         # Get soil parameters from request
#         nitrogen = float(request.args.get('N'))
#         phosphorus = float(request.args.get('P'))
#         potassium = float(request.args.get('K'))
#         ph = float(request.args.get('ph'))
#         ec = float(request.args.get('EC'))
#
#         # Make prediction
#         prediction = model.predict([[nitrogen, phosphorus, potassium, ph, ec]])
#         predicted_crop = prediction[0]  # Get the first predicted crop
#
#         return jsonify({"predicted_crop": predicted_crop})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400
#
#
# if __name__ == '_main_':
#     app.run(debug=True)