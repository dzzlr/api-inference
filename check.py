import requests
import json

response = requests.get(
    'http://127.0.0.1:5000/', 
    data = json.dumps({"query": "It was awesome movie"}),
    headers = {"Content-Type": "application/json"}
)

print(response.json())

