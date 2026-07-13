from groq import Groq

from config.settings import settings


class GroqClient:

    def __init__(self):
        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    def ask(self, question: str) -> str:

        response = self.client.chat.completions.create(

            model=settings.MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=settings.TEMPERATURE,

            max_completion_tokens=settings.MAX_TOKENS

        )

        return response.choices[0].message.content


groq_client = GroqClient()