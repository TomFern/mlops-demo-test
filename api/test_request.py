import requests

url = 'http://127.0.0.1:8080/predict'  # Replace with the URL you want to send the POST request to

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
