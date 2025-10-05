# Crop Recommendation System

A smart crop recommendation system that analyzes soil parameters and weather data to suggest the most suitable crops for your agricultural land.

## Features

- *Soil Analysis*: Upload soil images for automated analysis of soil parameters (N, P, K, pH)
- *Weather Integration*: Automatically fetches local weather data for your location
- *Crop Recommendation*: Uses machine learning to recommend optimal crops based on soil and weather conditions
- *User-Friendly Interface*: Modern web interface with responsive design

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. Clone this repository:

   
   git clone <repository-url>
   cd crop-recommendation-system
   

2. Run the application:

   Add API Keys(.env, src/app.py(Line 17 and Line 994))
   python run_app.py
   

   This script will:

   - Install all required dependencies
   - Train/download machine learning models if needed
   - Start the web server
   - Open the application in your default web browser

## Usage

1. *Start the Application*:

   
   python run_app.py
   

2. *Analyze Soil*:

   - Click on "Get Recommendation" in the navigation bar
   - Upload a soil image using the file input(use soilphotos images )
   - Click "Analyze Soil" to process the image
   - View the extracted soil parameters (N, P, K, pH)

3. *Get Weather Data*:

   - Click the "Weather" button to fetch current weather data for your location
   - Allow location access when prompted

4. *Predict Crop*:
   - After soil analysis and weather data are loaded
   - Adjust pH and rainfall values if needed
   - Click "Predict Crop" to get the recommended crop

## Model Training

The system uses several machine learning models:

1. *Soil Image Analysis Model*: Extracts soil parameters from images
2. *Crop Recommendation Model*: Predicts suitable crops based on soil and weather parameters

To retrain models with new data:


python train_model.py


## Data Sources

- *Crop Recommendation Data*: data/crop_recommendation.csv
- *Soil Parameters Data*: data/random_merged_soil_dataset.csv
- *Soil Images*: data/soil/ directory (extracted from soil.zip)

## System Architecture

The application consists of:

- *Flask Web Server*: Handles HTTP requests and serves the web interface
- *Image Processing Pipeline*: Extracts features from soil images
- *Machine Learning Models*: Predict soil parameters and recommend crops
- *Weather API Integration*: Fetches real-time weather data
- *User Interface*: HTML/CSS/JavaScript frontend

## Improving Model Accuracy

Current model accuracy can be improved by:

1. *Better Data Quality*: Ensure soil images are properly labeled with accurate soil parameters
2. *Feature Engineering*: The improved feature extraction includes color, texture, and edge features
3. *Advanced Models*: The system now uses ensemble methods (GradientBoosting) with hyperparameter tuning
4. *Data Augmentation*: Apply transformations to increase training data variety

## Troubleshooting

- *Missing Models*: Run python train_model.py to train all models
- *Weather Data Error*: Check internet connection and allow location access
- *Soil Analysis Error*: Ensure image is clear and well-lit
- *Database Connection Error*: Check MySQL configuration in src/app.py

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Weather data provided by OpenWeatherMap API
- Soil dataset compiled from various agricultural resources
- Crop recommendation dataset from Kaggle