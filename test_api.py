import requests

url = "http://127.0.0.1:8081/upload"

with open("test.txt", "rb") as f:
    files = {"file": f}
    data = {"job_description": "We need python, java, sql, nodejs, aws"}
    
    response = requests.post(url, files=files, data=data)
    
    print(response.json())
