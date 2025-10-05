"""
Complete Setup and Run Script for Crop Recommendation System

This script handles:
1. Installation of all required dependencies
2. Setting up the environment
3. Training models if necessary
4. Starting the application
"""

import os
import sys
import subprocess
import platform
import time
import webbrowser

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m' if platform.system() != 'Windows' else ''
    YELLOW = '\033[93m' if platform.system() != 'Windows' else ''
    RED = '\033[91m' if platform.system() != 'Windows' else ''
    BLUE = '\033[94m' if platform.system() != 'Windows' else ''
    ENDC = '\033[0m' if platform.system() != 'Windows' else ''

def print_banner():
    """Print a fancy banner for the application"""
    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}CROP RECOMMENDATION SYSTEM{Colors.ENDC}")
    print("=" * 80)
    print("A smart crop advisory system using soil image analysis and machine learning")
    print()

def print_section(title):
    """Print a section title"""
    print(f"\n{Colors.BLUE}> {title}{Colors.ENDC}")
    print("-" * 40)

def run_command(command, description=None, exit_on_error=True):
    """Run a shell command with proper error handling"""
    if description:
        print(f"  {description}... ", end='', flush=True)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True)
        if description:
            print(f"{Colors.GREEN}Done{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        if description:
            print(f"{Colors.RED}Failed{Colors.ENDC}")
        print(f"  {Colors.RED}Error:{Colors.ENDC} {e}")
        print(f"  {Colors.RED}Details:{Colors.ENDC} {e.stderr}")
        if exit_on_error:
            print(f"\n{Colors.RED}Setup failed. Please fix the errors and try again.{Colors.ENDC}")
            sys.exit(1)
        return False

def install_dependencies():
    """Install all required Python packages"""
    print_section("Installing Dependencies")
    
    dependencies = [
        "flask==2.0.1",
        "numpy==1.22.4",
        "pandas==1.3.3",
        "scikit-learn==1.0",
        "opencv-python==4.5.3.56",
        "scikit-image==0.18.3",
        "joblib==1.1.0",
        "matplotlib==3.4.3",
        "openpyxl==3.0.9",
        "requests==2.26.0",
        "Werkzeug==2.0.1"
    ]
    
    for package in dependencies:
        name = package.split('==')[0]
        version = package.split('==')[1]
        run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {name} {version}",
            exit_on_error=False
        )

def check_directories():
    """Ensure all necessary directories exist"""
    print_section("Checking Directories")
    
    directories = [
        "src/models",
        "src/static",
        "src/static/images",
        "src/uploads",
        "data"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"  Creating {directory}...")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"  ✓ {directory} exists")

def check_and_train_models():
    """Check if models exist and train them if needed"""
    print_section("Checking Models")
    
    model_files = [
        "src/models/crop_recommendation_model.pkl",
        "src/models/label_encoder.pkl",
        "src/models/image_classifier_model.pkl",
        "src/models/soil_params_model.pkl"
    ]
    
    models_exist = True
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"  {Colors.YELLOW}Missing:{Colors.ENDC} {model_file}")
            models_exist = False
        else:
            print(f"  ✓ {model_file} exists")
    
    if not models_exist:
        print_section("Training Models")
        run_command(f"{sys.executable} train_model.py", "Training machine learning models")

def start_application():
    """Start the Flask application and open browser"""
    print_section("Starting Application")
    
    # Create a process for the Flask app
    try:
        if platform.system() == 'Windows':
            process = subprocess.Popen(
                [sys.executable, 'src/app.py'],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                [sys.executable, 'src/app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        print(f"  Application starting on {Colors.GREEN}http://localhost:5000{Colors.ENDC}")
        print(f"  {Colors.YELLOW}Please wait...{Colors.ENDC}")
        
        # Wait for the server to start
        time.sleep(3)
        
        # Open in browser
        print(f"  Opening browser...")
        webbrowser.open('http://localhost:5000')
        
        print("\n" + "=" * 80)
        print(f"{Colors.GREEN}Crop Recommendation System is now running!{Colors.ENDC}")
        print(f"Access the web interface at: {Colors.GREEN}http://localhost:5000{Colors.ENDC}")
        print(f"Press {Colors.YELLOW}Ctrl+C{Colors.ENDC} to stop the application")
        print("=" * 80 + "\n")
        
        # Keep the process running until user interrupts
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print(f"{Colors.YELLOW}Stopping the application...{Colors.ENDC}")
        process.terminate()
        print(f"{Colors.GREEN}Application stopped successfully.{Colors.ENDC}")
        print("=" * 80)
    except Exception as e:
        print(f"\n{Colors.RED}Error starting application:{Colors.ENDC} {e}")
        sys.exit(1)

def main():
    """Main function to run the setup and start the application"""
    print_banner()
    
    # Install dependencies
    install_dependencies()
    
    # Check and create directories
    check_directories()
    
    # Check and train models if needed
    check_and_train_models()
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()