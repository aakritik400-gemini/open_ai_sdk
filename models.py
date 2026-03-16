import requests
import os

API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

response = requests.get(url)

data = response.json()

for model in data.get("models", []):
    print(model["name"])