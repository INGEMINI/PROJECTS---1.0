import requests

# Replace with your OpenWeatherMap API key
api_key = "my api key "

# Prompt user for city name
city = input("Enter city: ")

# Make a request to the OpenWeatherMap API
response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}")

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON data from the response
    data = response.json()

    # Extract and display the weather description
    weather_description = data["weather"][0]["description"]
    print(f"Weather in {city}: {weather_description}")

else:
    # Display an error message if the request failed
    print(f"Error fetching data. Status Code: {response.status_code}")
    print("Response:", response.json())
