import os
from typing import Optional, List, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
]


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
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Instantiate a ChatGoogleGenerativeAI model with automatic quota/rate-limit fallbacks
    across available Gemini models.
    """
    key = api_key or get_google_api_key()
    primary_model = model or os.getenv("GEMINI_MODEL", FALLBACK_MODELS[0])

    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    instances = [
        ChatGoogleGenerativeAI(model=m, google_api_key=key, temperature=temperature)
        for m in models_to_try
    ]

    primary = instances[0]
    if len(instances) > 1:
        return primary.with_fallbacks(instances[1:])
    return primary


def get_structured_llm(
    schema: Any,
    temperature: float = 0.0,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Create a structured output LLM chain with automatic fallbacks to ensure
    resilience against per-model free tier quotas.
    """
    key = api_key or get_google_api_key()
    primary_model = model or os.getenv("GEMINI_MODEL", FALLBACK_MODELS[0])

    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    structured_instances = [
        ChatGoogleGenerativeAI(
            model=m, google_api_key=key, temperature=temperature
        ).with_structured_output(schema)
        for m in models_to_try
    ]

    primary = structured_instances[0]
    if len(structured_instances) > 1:
        return primary.with_fallbacks(structured_instances[1:])
    return primary
