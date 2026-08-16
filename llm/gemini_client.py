from google import genai

from config.settings import settings


class GeminiClient:

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

    def ask(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        return response.text.strip()


gemini_client = GeminiClient()