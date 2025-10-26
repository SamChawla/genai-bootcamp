# main.py
from app_config import env_config

# Groq SDK (optional; will error if not installed and Groq models are selected)
from groq import Groq

# New OpenAI client (requires openai>=1.0.0)
from openai import OpenAI

# Bot identity and default system prompt
BOT_NAME = env_config.default_bot_name
DEFAULT_SYSTEM_PROMPT = env_config.default_system_prompt

# Models that should be routed to OpenAI
OPENAI_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}

class LLMApp:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        """
        Initialize LLMApp with either Groq or OpenAI client depending on the selected model.
        api_key: optional API key to use (overrides env-config)
        model: model name string
        """
        self.model = model
        self.conversation_history = []

        # Decide provider
        self.provider = "openai" if self.model in OPENAI_MODELS else "groq"

        # Pick API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = env_config.openai_api_key if self.provider == "openai" else env_config.groq_api_key

        if not self.api_key:
            raise ValueError(
                f"API key for provider '{self.provider}' is required. "
                f"Provide it via the `api_key` param or set the appropriate environment variable."
            )

        # Initialize client for chosen provider
        if self.provider == "openai":
            # New OpenAI client (openai>=1.0.0)
            self.openai_client = OpenAI(api_key=self.api_key)
            self.groq_client = None
        else:
            if Groq is None:
                raise RuntimeError("Groq SDK is not installed; install it to use Groq models.")
            self.groq_client = Groq(api_key=self.api_key)
            self.openai_client = None

    def _build_system_message(self, user_system_prompt: str | None):
        base_prompt = user_system_prompt.strip() if user_system_prompt and user_system_prompt.strip() else DEFAULT_SYSTEM_PROMPT
        persona_prompt = f"You are {BOT_NAME}. {base_prompt}"
        return {"role": "system", "content": persona_prompt}

    def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
        # Compose messages: system -> history -> user
        messages = [self._build_system_message(system_prompt)]
        if self.conversation_history:
            messages.extend(self.conversation_history)

        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        self.conversation_history.append(user_msg)

        if self.provider == "openai":
            # Use new OpenAI client parameter name: max_completion_tokens
            resp = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            assistant_text = resp.choices[0].message.content
        else:
            # Groq client uses the older 'max_tokens' parameter
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


if __name__ == "__main__":
    # Basic CLI test (will require env keys or param)
    try:
        app = LLMApp()
        q = input("Ask: ")
        print(app.chat(q))
    except Exception as e:
        print("Error:", e)