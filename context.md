# Crop Recommendation System Documentation

## Overview

This project is an AI-powered crop recommendation system that analyzes soil parameters and climate data to suggest the most suitable crops for farming. The system uses machine learning models trained on agricultural datasets to make accurate predictions.

## Features

- **Soil Analysis**: Upload soil images to get NPK and pH values
- **Weather Integration**: Automatically fetches temperature and humidity data based on location
- **Crop Prediction**: Recommends optimal crops based on soil and climate conditions
- **User Authentication**: Secure login and signup functionality

## Technical Architecture

The application follows a client-server architecture:

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Database**: MySQL
- **ML Models**: Scikit-learn (RandomForestClassifier)

## Machine Learning Models

The system uses several trained models:

1. **Soil Model**: Analyzes soil images to extract NPK and pH values
2. **Crop Recommendation Model**: Predicts suitable crops based on soil parameters and climate data

## API Endpoints

- `/analyze-soil`: Processes soil images to extract parameters
- `/predict`: Recommends crops based on input parameters
- `/get_weather`: Fetches weather data for a location
- `/get_soil_data`: Retrieves soil data for specific districts

## Data Processing

The system processes the following data:

- **Soil Parameters**: Nitrogen (N), Phosphorus (P), Potassium (K), pH
- **Climate Data**: Temperature, Humidity, Rainfall
- **Image Features**: Color histograms, texture features from soil images

## Fallback Mechanisms

The prediction system includes multiple fallback strategies:

1. Primary: Model-based prediction using different feature combinations
2. Secondary: Nearest-neighbor analysis comparing input values to crop profiles
3. Tertiary: Rule-based recommendations for exceptional cases

## Usage Instructions

1. Register or login to the system
2. Upload a soil image for analysis
3. The system will extract soil parameters (N, P, K, pH)
4. Allow location access for weather data
5. Click "Predict Crop" to get recommendations

## Development Notes

- Environmental variables are stored in `.env`
- Models are saved in `src/models/`
- Image uploads are stored temporarily in `src/uploads/`
- Database schema is defined in the MySQL database `agriculture_db`
