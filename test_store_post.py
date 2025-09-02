import requests
import time

payload = {
    'user_id': 'test',
    'post_id': 'test_post_1', 
    'content': 'This is a test post',
    'timestamp': int(time.time()),
    'tags': ['test']
}

print('Sending payload:', payload)

try:
    response = requests.post('http://localhost:8001/store-post', json=payload)
    print('Status:', response.status_code)
    print('Response:', response.text)
except Exception as e:
    print('Error:', e)
