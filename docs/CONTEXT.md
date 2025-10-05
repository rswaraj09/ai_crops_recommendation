# Crop Recommendation System

## Project Overview

This project is an intelligent crop recommendation system that uses machine learning to suggest suitable crops based on soil characteristics, weather conditions, and geographical location. The system helps farmers make data-driven decisions about what crops to plant to maximize yield and profit.

## Key Components

### 1. Data Sources

- **Crop Recommendation Dataset** (`data/crop_recommendation.csv`): Contains soil parameters (N, P, K, pH), climate data (temperature, humidity, rainfall), and corresponding crop recommendations.
- **Soil Dataset** (`data/random_merged_soil_dataset.csv`): Contains detailed soil analysis data by district, including micronutrients (Zn, Fe, Cu, Mn, B, S) and macronutrients (N, P, K).
- **Market Data** (`data/market.csv`): Contains agricultural commodity prices across different markets to help with economic decision-making.
- **Crops Dataset** (`data/Crops.csv`): Detailed information about various crops including cultivation requirements and expected yields.
- **Image Dataset** (`data/imagedatasets1.xlsx`): Contains crop and soil image data with corresponding features for training visual recognition models.
- **Soil Images** (`data/soil.zip`): A collection of soil images used for training and testing the soil analysis model.

### 2. Machine Learning Models

- **Crop Recommendation Model**: A Random Forest classifier trained on soil and climate parameters to predict the most suitable crop.
- **Soil Analysis Model**: Uses computer vision techniques to analyze soil images and extract features.
- **Image-Based Crop and Soil Classifier**: A scikit-learn model trained on the imagedatasets1 dataset to identify crop types and soil conditions from images.

### 3. Web Application

The system is implemented as a Flask web application with features including:

- Soil parameter input (manual or by selecting a district)
- Weather data fetching using OpenWeatherMap API
- Crop recommendation based on input parameters
- Soil image analysis for nutrient prediction
- Image-based crop and soil identification

## Code Structure

- **app.py**: Main Flask application that handles routing, API endpoints, and database connections
- **train_model.py**: Scripts for training the crop recommendation model
- **predict_crop.py**: Functions to predict crops based on input parameters
- **soil_model.py**: Model for soil analysis and feature extraction
- **region.py**: API endpoints for district-based soil data retrieval
- **templates/**: HTML templates for the web interface
- **train_image_model.py**: Script for training the image-based soil analysis model

## How It Works

1. **Data Collection**: The system either uses pre-existing soil data based on the district or allows users to input soil parameters manually. Weather data is fetched in real-time using geolocation. Users can also upload images of crops or soil for analysis.

2. **Data Processing**: The collected data is processed and formatted for the prediction model.

3. **Crop Prediction**: The machine learning model analyzes the soil and climate data to recommend the most suitable crop.

4. **Image Analysis**: For uploaded images, the image classification model identifies the crop type or soil condition and extracts relevant features.

5. **Results Display**: The system displays the recommended crop along with cultivation tips and market information.

## Database Schema

The system uses a MySQL database with the following tables:

- Users: Stores user credentials and profiles
- SoilData: Stores soil analysis results by district
- CropRecommendations: Stores historical recommendations
- ImageAnalysis: Stores results from image-based predictions

## API Endpoints

- `/get_soil_data`: Returns soil parameters for a given district
- `/predict_crop`: Predicts the best crop based on soil and weather parameters
- `/get_weather`: Fetches weather data based on geolocation
- `/analyze-soil`: Analyzes soil images to extract parameters
- `/analyze-image`: Analyzes crop or soil images for identification and feature extraction

## Technologies Used

- **Backend**: Python, Flask
- **Database**: MySQL
- **Machine Learning**: Scikit-learn, OpenCV
- **Deep Learning**: TensorFlow/PyTorch for image classification
- **APIs**: OpenWeatherMap
- **Frontend**: HTML, CSS, JavaScript, Bootstrap

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up the database: Create a MySQL database called `agriculture_db`
3. Run the application: `python src/app.py`
4. Access the web interface at: `http://localhost:5000`

## Image-Based Soil Analysis

The system includes a soil image analysis component that uses machine learning to extract soil parameters from uploaded images:

1. **Image Processing**: The uploaded soil image is processed using OpenCV to resize and normalize the image.
2. **Feature Extraction**: Color histograms and statistical features are extracted from the image.
3. **Parameter Prediction**: The trained machine learning model analyzes these features to predict soil parameters.
4. **Results Display**: The predicted soil parameters (N, P, K, pH, rainfall) are displayed to the user.

### Training the Soil Image Analysis Model

The soil image analysis model is trained using the script `train_image_model.py`:

1. The script loads soil images from the `data/soil.zip` archive.
2. Features are extracted from each image using OpenCV.
3. A RandomForest classifier is trained with hyperparameter tuning via GridSearchCV.
4. The trained model is saved as `models/image_classifier_model.pkl`.

To train the model, run:

```bash
python src/train_image_model.py
```

### Using the Soil Image Analysis Feature

1. On the web interface, navigate to the "Upload Soil Image for Analysis" section.
2. Upload a soil image using the file picker.
3. Click "Analyze Soil" to process the image.
4. The predicted soil parameters will be displayed.
5. Click "Predict Crop" to get a crop recommendation based on these parameters.
