# main.py
from app_config import env_config

# optional imports
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

BOT_NAME = env_config.default_bot_name if env_config else "Bot"
DEFAULT_SYSTEM_PROMPT = env_config.default_system_prompt if env_config else "You are a helpful assistant."

OPENAI_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4o", "gpt-4"}

class LLMApp:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        self.model = model
        self.conversation_history = []
        self.provider = "openai" if self.model in OPENAI_MODELS else "groq"

        if api_key:
            self.api_key = api_key
        else:
            if env_config:
                self.api_key = env_config.openai_api_key if self.provider == "openai" else env_config.groq_api_key
            else:
                self.api_key = None

        if not self.api_key:
            raise ValueError(f"API key for provider '{self.provider}' is required. Provide via api_key or env var.")

        if self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("OpenAI client not installed.")
            self.openai_client = OpenAI(api_key=self.api_key)
            self.groq_client = None
        else:
            if Groq is None:
                raise RuntimeError("Groq client not installed.")
            self.groq_client = Groq(api_key=self.api_key)
            self.openai_client = None

    def _build_system_message(self, user_system_prompt: str | None):
        base_prompt = user_system_prompt.strip() if user_system_prompt and user_system_prompt.strip() else DEFAULT_SYSTEM_PROMPT
        persona_prompt = f"You are {BOT_NAME}. {base_prompt}"
        return {"role": "system", "content": persona_prompt}

    def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
        messages = [self._build_system_message(system_prompt)]
        if self.conversation_history:
            messages.extend(self.conversation_history)

        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        self.conversation_history.append(user_msg)

        if self.provider == "openai":
            resp = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            assistant_text = resp.choices[0].message.content
        else:
            resp = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            assistant_text = resp.choices[0].message.content

        assistant_msg = {"role": "assistant", "content": assistant_text}
        self.conversation_history.append(assistant_msg)
        return assistant_text

    def clear_history(self):
        self.conversation_history = []

    def get_history(self):
        return self.conversation_history
