from memory.memory_api import MemoryAPIClient
import time

client = MemoryAPIClient()
result = client.store_post(
    user_id='default',
    post_id='test_fb_post_1',
    content='This is a test Facebook post',
    timestamp=int(time.time()),
    tags=['facebook', 'test']
)
print('Store result:', result)
