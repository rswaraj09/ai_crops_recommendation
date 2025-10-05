import os
import sys
import subprocess
import webbrowser
import time
import threading

def check_models():
    """Check if required models exist, and train them if they don't"""
    models_dir = os.path.join('src', 'models')
    required_models = [
        'crop_recommendation_model.pkl',
        'label_encoder.pkl',
        'image_classifier_model.pkl',
        'soil_model.pkl'
    ]
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        print("Created models directory")
    
    # Check if models exist
    missing_models = [model for model in required_models 
                      if not os.path.exists(os.path.join(models_dir, model))]
    
    if missing_models:
        print(f"Missing required models: {', '.join(missing_models)}")
        print("Running model training script...")
        subprocess.run([sys.executable, 'train_model.py'])
    else:
        print("All required models are present")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'numpy', 'pandas', 'scikit-learn', 'opencv-python', 'joblib', 'flask_cors'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("All dependencies installed")

def run_app():
    """Run the Flask application"""
    print("Starting the Crop Recommendation System...")
    
    # Run the application directly instead of using a subprocess
    print("Starting Flask server directly. Please wait...")
    print("The web browser will open automatically when the server is ready.")
    print("\n" + "="*80)
    print("Crop Recommendation System is now running!")
    print("Access the application at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    # Open browser in a separate thread after a delay
    def open_browser():
        time.sleep(3)  # Wait for server to start
        webbrowser.open('http://127.0.0.1:5000')
    
    # Start browser opener thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the app directly
    os.system("python src/app.py")

if __name__ == '__main__':
    print("="*80)
    print("Crop Recommendation System")
    print("="*80)
    
    # Check dependencies
    check_dependencies()
    
    # Check and train models if needed
    check_models()
    
    # Run the application
    run_app()