import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

LLM_CONFIG = {
    "config_list": [
        {
            "model": "gemini-3-flash-preview",
            "api_key": GEMINI_API_KEY,
            "api_type": "google",
        }
    ],
    "temperature": 0.2,
}
