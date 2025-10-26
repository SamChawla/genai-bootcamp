# appconfig.py
import os
from dotenv import load_dotenv, find_dotenv

# Load .env if present (safe to call multiple times)
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

class EnvConfig:
    @property
    def groq_api_key(self):
        return os.getenv("GROQ_API_KEY")

    @property
    def openai_api_key(self):
        return os.getenv("OPENAI_API_KEY")

    @property
    def default_bot_name(self):
        return os.getenv("DEFAULT_BOT_NAME", "BotMan")

    @property
    def default_system_prompt(self):
        # Short default prompt; can be overridden via env or UI
        return os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are a helpful, concise, and accurate question-answering assistant. "
            "Be polite and prefer clarity. If you don't know an answer, say so."
        )

env_config = EnvConfig()