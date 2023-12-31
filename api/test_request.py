import requests
import os
import sys

# To change the test url, set the env variable TEST_URL
# e.g. export TEST_URL='https://mlops-demo-test-purple-river-469.fly.dev/predict'
default_url = 'http://127.0.0.1:8080/predict'
url = os.environ.get('TEST_URL', default_url)

data = {
        'MedInc': 3.5,        # Median income in tens of thousands
        'HouseAge': 35,       # House age in years
        'AveRooms': 6,        # Average number of rooms
        'AveBedrms': 2,       # Average number of bedrooms
        'Population': 800,    # Population in the block
        'AveOccup': 3,        # Average occupancy per household
        'Latitude': 34.2,     # Latitude of the block
        'Longitude': -118.4   # Longitude of the block
}

# Optional: Add headers, if required by the API
headers = {
    'Content-Type': 'application/json'
    # Add other headers as required
}

# Make the POST request
response = requests.post(url, json=data, headers=headers)

# Check the response
print(f'Status Code: {response.status_code}')
print(f'Response Body: {response.text}')
if response.status_code != 200:
    sys.exit(1)
