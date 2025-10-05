"""
Test the soil analysis API endpoint

This script tests the functionality of the /analyze-soil API endpoint
by sending a sample image and verifying the response format.
"""

import os
import requests
import sys
import numpy as np
from PIL import Image
import time
import platform

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m' if platform.system() != 'Windows' else ''
    YELLOW = '\033[93m' if platform.system() != 'Windows' else ''
    RED = '\033[91m' if platform.system() != 'Windows' else ''
    BLUE = '\033[94m' if platform.system() != 'Windows' else ''
    ENDC = '\033[0m' if platform.system() != 'Windows' else ''

def create_test_image():
    """Create a test soil image if no sample is available"""
    print(f"{Colors.YELLOW}Creating a test soil image...{Colors.ENDC}")
    
    # Create test directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # Create a simple brown soil-like image
    width, height = 200, 200
    img = Image.new('RGB', (width, height), color=(139, 69, 19))  # Brown color
    
    # Add some texture and variations to make it look more like soil
    pixels = np.array(img)
    for i in range(width):
        for j in range(height):
            # Add random variations to create soil texture
            r = min(255, max(0, pixels[i, j, 0] + np.random.randint(-30, 30)))
            g = min(255, max(0, pixels[i, j, 1] + np.random.randint(-30, 30)))
            b = min(255, max(0, pixels[i, j, 2] + np.random.randint(-20, 20)))
            pixels[i, j] = [r, g, b]
    
    # Convert back to PIL Image and save
    img = Image.fromarray(pixels)
    test_image_path = os.path.join("test_images", "test_soil.jpg")
    img.save(test_image_path)
    
    print(f"Test image created at: {test_image_path}")
    return test_image_path

def find_sample_image():
    """Find a sample soil image for testing"""
    # Check in data/soil directory
    soil_dir = os.path.join("data", "soil")
    if os.path.exists(soil_dir):
        for root, _, files in os.walk(soil_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(root, file)
    
    # Check in src/static/images
    images_dir = os.path.join("src", "static", "images")
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(images_dir, file)
    
    # No suitable image found, create one
    return create_test_image()

def test_soil_analysis_api(server_url="http://localhost:5000"):
    """Test the soil analysis API endpoint"""
    print("\n" + "=" * 80)
    print(f"{Colors.BLUE}TESTING SOIL ANALYSIS API{Colors.ENDC}")
    print("=" * 80 + "\n")
    
    # Find a sample image to use for testing
    sample_image_path = find_sample_image()
    print(f"Using sample image: {sample_image_path}")
    
    # Check if the server is running
    try:
        print(f"Checking if server is running at {server_url}...")
        response = requests.get(server_url, timeout=5)
        if response.status_code != 200:
            print(f"{Colors.YELLOW}Server returned status code {response.status_code}{Colors.ENDC}")
            print("Starting server...")
            
            # Try to start the server
            import subprocess
            subprocess.Popen([sys.executable, 'src/app.py'])
            print("Waiting for server to start...")
            time.sleep(5)  # Wait for server to start
    except requests.exceptions.RequestException:
        print(f"{Colors.YELLOW}Server is not running at {server_url}{Colors.ENDC}")
        print("Starting server...")
        
        # Try to start the server
        import subprocess
        subprocess.Popen([sys.executable, 'src/app.py'])
        print("Waiting for server to start...")
        time.sleep(5)  # Wait for server to start
    
    # Test the API endpoint
    api_url = f"{server_url}/analyze-soil"
    print(f"\nTesting API endpoint: {api_url}")
    
    try:
        # Prepare the file for upload
        with open(sample_image_path, 'rb') as img_file:
            files = {'soil_image': (os.path.basename(sample_image_path), img_file, 'image/jpeg')}
            
            # Send the request
            print("Sending request with sample image...")
            response = requests.post(api_url, files=files, timeout=30)
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                print(f"\n{Colors.GREEN}SUCCESS!{Colors.ENDC} API endpoint is working correctly.")
                print("\nAPI Response:")
                print(f"  Nitrogen (N): {result.get('nitrogen')}")
                print(f"  Phosphorus (P): {result.get('phosphorus')}")
                print(f"  Potassium (K): {result.get('potassium')}")
                print(f"  pH Level: {result.get('ph')}")
                
                # Validate response structure
                expected_keys = ['nitrogen', 'phosphorus', 'potassium', 'ph']
                missing_keys = [key for key in expected_keys if key not in result]
                
                if missing_keys:
                    print(f"\n{Colors.YELLOW}WARNING:{Colors.ENDC} Some expected keys are missing: {', '.join(missing_keys)}")
                else:
                    print(f"\n{Colors.GREEN}All expected response fields are present.{Colors.ENDC}")
                
                return True
            else:
                print(f"\n{Colors.RED}ERROR:{Colors.ENDC} API returned status code {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error message: {error_data.get('error', 'No error message provided')}")
                except:
                    print(f"Response text: {response.text}")
                return False
    except Exception as e:
        print(f"\n{Colors.RED}ERROR:{Colors.ENDC} Failed to test API: {e}")
        return False

if __name__ == "__main__":
    # Test the API
    success = test_soil_analysis_api()
    
    # Exit with appropriate status code
    if success:
        print(f"\n{Colors.GREEN}API TEST PASSED{Colors.ENDC}")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}API TEST FAILED{Colors.ENDC}")
        sys.exit(1) 