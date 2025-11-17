# src/llms/groq_client.py
"""
Groq LLM client wrapper — robust to differences between groq SDK versions.

This wrapper:
 - Safely handles missing `groq.errors` module by falling back to Exception.
 - Tries to instantiate the Groq client even if the SDK signature differs.
 - Extracts text from a variety of response shapes.
"""

from __future__ import annotations
import os
import time
import logging
from typing import Optional, Any

# Try to import the Groq SDK. If not installed, imports will raise ImportError
try:
    from groq import Groq  # type: ignore
except Exception as exc:  # pragma: no cover - defensive import
    Groq = None  # type: ignore

# Try to import SDK error class if available; otherwise fall back to Exception
try:
    # Some versions expose an `errors` module with GroqError
    from groq.errors import GroqError  # type: ignore
except Exception:  # pragma: no cover - defensive import
    # Try alternative names or fall back
    try:
        GroqError = getattr(__import__("groq"), "GroqError")  # type: ignore
    except Exception:
        GroqError = Exception  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GroqClientError(RuntimeError):
    pass


class GroqClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "llama-3.1-8b-instant",
        timeout: int = 120,
    ) -> None:
        """
        Minimal wrapper around Groq Python SDK.

        :param api_key: optional API key (if not provided we'll pick from GROQ_API_KEY env var)
        :param default_model: model name to use (Groq model strings, e.g. "llama-3.1-8b-instant")
        :param timeout: timeout for SDK calls (seconds)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = default_model
        self.timeout = timeout
        self._client = None

        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. Provide it via environment or .env file.")

        if Groq is None:
            logger.warning(
                "Groq SDK not available (import failed). Install the official SDK with `pip install groq` "
                "or check your environment."
            )
            # Don't raise here — allow code to import the module so Streamlit can show an error in UI later.
            return

        # Instantiate the SDK client in a defensive way (different SDK versions differ in constructor)
        try:
            # Preferred: Groq(api_key=..., timeout=...)
            try:
                self._client = Groq(api_key=self.api_key, timeout=self.timeout) if self.api_key else Groq(timeout=self.timeout)
            except TypeError:
                # Fallback: maybe Groq(...) expects no keyword args or different names
                try:
                    self._client = Groq(self.api_key, self.timeout) if self.api_key else Groq()
                except Exception:
                    # Final fallback: attempt no-arg instantiation and hope it reads env
                    self._client = Groq()
        except Exception as exc:
            logger.exception("Failed to instantiate Groq client: %s", exc)
            raise GroqClientError(f"Failed to create Groq client: {exc}") from exc

    def set_model(self, model_name: str) -> None:
        self.model = model_name
        logger.debug("GroqClient model set to %s", model_name)

    def _extract_text_from_response(self, resp: Any) -> str:
        """
        Extract returned text from Groq chat completion response.
        Handles multiple shapes (dict, object, or simple string).
        """
        # Quick sanity
        if resp is None:
            return ""

        # If it's already a plain string, return it
        if isinstance(resp, str):
            return resp.strip()

        # If dict-like
        if isinstance(resp, dict):
            # OpenAI-like: {"choices":[{"message":{"content":"..."}}]}
            choices = resp.get("choices")
            if choices and isinstance(choices, (list, tuple)):
                first = choices[0]
                if isinstance(first, dict):
                    # message.content
                    msg = first.get("message") or first.get("delta") or first.get("text")
                    if isinstance(msg, dict) and "content" in msg:
                        return str(msg["content"]).strip()
                    if isinstance(msg, str):
                        return msg.strip()
                    # sometimes content is under 'text'
                    if "text" in first and isinstance(first["text"], str):
                        return first["text"].strip()
            # sometimes key 'text' exists at top-level
            if "text" in resp and isinstance(resp["text"], str):
                return resp["text"].strip()

        # If object-like with attributes (choices -> message -> content)
        try:
            choices = getattr(resp, "choices", None)
            if choices:
                first = choices[0]
                # message.content
                message = getattr(first, "message", None)
                if message:
                    content = getattr(message, "content", None)
                    if content:
                        return str(content).strip()
                # fallback to text attribute on choice
                text_attr = getattr(first, "text", None)
                if text_attr:
                    return str(text_attr).strip()
        except Exception:
            logger.debug("Non-fatal: attribute-style extraction failed", exc_info=True)

        # Last resort: attempt to stringify a nested common path or the whole object
        try:
            # try common nested keys if dict-like and not yet handled
            if isinstance(resp, dict):
                # flatten simple nested dict values to strings
                for k in ("output", "result", "response"):
                    if k in resp and isinstance(resp[k], str):
                        return resp[k].strip()
            return str(resp).strip()
        except Exception as exc:
            logger.exception("Unable to extract text from Groq response: %s", exc)
            raise GroqClientError("Unable to parse Groq response") from exc

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, system_prompt: Optional[str] = None, retries: int = 2, **kwargs
    ) -> str:
        """
        Create a chat completion using the Groq SDK (defensive mode).

        :param prompt: user prompt string
        :param max_tokens: approximate maximum tokens
        :param temperature: sampling temperature
        :param system_prompt: optional system instruction
        :param retries: number of retries on transient failures
        :returns: generated text
        """
        if Groq is None or self._client is None:
            raise GroqClientError("Groq SDK not available or client failed to initialize. Install/verify the `groq` package and GROQ_API_KEY.")

        if not self.api_key:
            raise GroqClientError("GROQ_API_KEY is missing. Set GROQ_API_KEY in the environment or pass api_key to GroqClient.")

        # Build messages in chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= retries:
            try:
                logger.debug(
                    "Groq chat completion call (model=%s, max_tokens=%s, temp=%s, attempt=%d)",
                    self.model,
                    max_tokens,
                    temperature,
                    attempt,
                )

                # Many versions of the SDK use this shape:
                # resp = client.chat.completions.create(model=..., messages=..., max_tokens=..., temperature=...)
                try:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                except AttributeError:
                    # Some SDK versions/surface area differ — try a slightly different call shape
                    # Maybe it's client.chat_create or client.completions.create etc.
                    try:
                        resp = self._client.chat.create(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
                    except Exception:
                        # Final fallback: try a top-level completions endpoint
                        resp = self._client.completions.create(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

                text = self._extract_text_from_response(resp)
                logger.debug("Extracted text length=%d", len(text))
                return text

            except GroqError as ge:
                # SDK-specific error — if auth-related, don't retry
                logger.exception("Groq SDK error on attempt %d: %s", attempt, ge)
                last_exc = ge
                status_code = getattr(ge, "status_code", None)
                if status_code in (401, 403):
                    raise GroqClientError("Authentication failed with Groq API. Check GROQ_API_KEY and permissions.") from ge
                attempt += 1
                time.sleep(1 + attempt * 0.5)
            except Exception as exc:
                logger.exception("Error calling Groq (attempt %d): %s", attempt, exc)
                last_exc = exc
                attempt += 1
                time.sleep(1 + attempt * 0.5)

        raise GroqClientError(f"Failed to generate text from Groq after {retries + 1} attempts: {last_exc}") from last_exc

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return str(text).strip()
