from groq import Groq

from config.settings import settings
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT


class GroqClient:

    def __init__(self):
        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    def _chat(self, messages: list[dict]) -> str:
        """
        Internal helper for sending chat requests to Groq.
        """

        response = self.client.chat.completions.create(

            model=settings.MODEL_NAME,

            messages=messages,

            temperature=settings.TEMPERATURE,

            max_completion_tokens=settings.MAX_TOKENS

        )

        return response.choices[0].message.content.strip()

    def ask(self, question: str) -> str:
        """
        General conversation with the LLM.
        """

        messages = [
            {
                "role": "user",
                "content": question
            }
        ]

        return self._chat(messages)

    def classify_route(self, question: str) -> str:
        """
        Ask the LLM which pipeline should handle the request.
        """

        messages = [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return self._chat(messages)


groq_client = GroqClient()