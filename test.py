import requests

# OpenRouter API key
OPENROUTER_API_KEY = "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"

# API endpoint for OpenRouter AI
url = "https://api.openrouter.ai/v1/completions"

# Headers including the authorization token
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# Data for the prompt and request (example)
data = {
    "model": "gpt-3.5-turbo",  # Replace with the model you're using
    "prompt": "Can you tell me about OpenRouter AI?",
    "max_tokens": 100
}

# Make the request to the OpenRouter AI API
response = requests.post(url, headers=headers, json=data)

# Check the response status and content
if response.status_code == 200:
    result = response.json()
    print("Response: ", result["choices"][0]["text"])  # Adjust based on actual response format
else:
    print(f"Failed to get response: {response.status_code}")
    print(f"Error: {response.text}")
