import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_google_api_key() -> str:
    key = (
        os.getenv("GOOGLEAI_STUDIO_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not key:
        raise ValueError(
            "Missing Google AI Studio / Gemini API key. Please ensure GOOGLEAI_STUDIO_KEY is set in .env."
        )
    return key


def get_llm(
    temperature: float = 0.0,
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    """Instantiate a ChatGoogleGenerativeAI model with standard parameters."""
    key = api_key or get_google_api_key()
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=temperature,
    )
