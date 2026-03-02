import requests

resp = requests.post('http://127.0.0.1:5000/predict', json={'text': 'Test uncertainty text'})
print(resp.status_code)
print(resp.json())
