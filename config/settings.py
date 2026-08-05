from pathlib import Path
from dotenv import load_dotenv
import os

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(PROJECT_ROOT / "config" / ".env")


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
    GOOGLE_EMAIL = os.getenv("GOOGLE_EMAIL")

settings = Settings()