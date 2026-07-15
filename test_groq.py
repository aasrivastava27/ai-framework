from dotenv import load_dotenv
from litellm import completion
import os

from dotenv import load_dotenv
load_dotenv("config/.env")

print("API KEY FOUND:", bool(os.getenv("GROQ_API_KEY")))

response = completion(
    model="groq/llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Say hello"
        }
    ]
)

print(response.choices[0].message.content)