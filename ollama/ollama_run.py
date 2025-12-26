import requests

url = "http://localhost:11434/v1/completions"

payload = {
    "model": "llama3.1:latest",   # your actual model
    "prompt": "Hello Ollama!"
}

response = requests.post(url, json=payload)
print(response.json())