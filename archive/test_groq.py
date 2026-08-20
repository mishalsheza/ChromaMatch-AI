# test_groq.py (in root folder)
from dotenv import load_dotenv
import os
from groq import Groq

# Load .env from backend folder
load_dotenv('backend/.env')

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

print("Testing qwen/qwen3.6-27b...")
response = client.chat.completions.create(
    model='qwen/qwen3.6-27b',
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Provide a direct, concise response without any thinking process."},
        {"role": "user", "content": "Say hello to someone with cool undertones in 10 words."}
    ],
    max_tokens=30,
    temperature=0.3
)

print("Response:", response.choices[0].message.content)

# Test with allam-2-7b
print("\nTesting allam-2-7b...")
response2 = client.chat.completions.create(
    model='allam-2-7b',
    messages=[
        {"role": "system", "content": "You are a friendly color analyst. Provide a direct, warm response."},
        {"role": "user", "content": "Give 3 colors for someone with cool undertones."}
    ],
    max_tokens=30
)

print("Response:", response2.choices[0].message.content)