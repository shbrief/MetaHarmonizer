"""Lightweight multi-provider LLM client.

Single-shot text generation against Anthropic Claude or Google Gemini.
Returns plain response text — callers handle prompt construction, parsing,
and retry/backoff strategies of their own.

Provider is selected by the model identifier prefix (``claude-*`` ->
Anthropic, ``gemini-*`` -> Gemini). Provider SDKs are optional and
imported lazily so that this module can be imported without either
``anthropic`` or ``google-generativeai`` installed; the import error is
raised only when the corresponding call is invoked.

Used by:
    - :mod:`metaharmonizer.models.schema_mapper.generate_alias_dict`
    - :class:`metaharmonizer.models.ontology_mapper_llm.OntoMapLLM`
"""
from __future__ import annotations

import os
from typing import Optional


DEFAULT_ANTHROPIC_TIMEOUT = 3600
DEFAULT_GEMINI_TIMEOUT = 1800
DEFAULT_GEMINI_TEMPERATURE = 0.2


def detect_provider(model: str) -> str:
    """Return the provider name implied by ``model``.

    ``claude-*`` -> ``"anthropic"``, ``gemini-*`` / ``gemma-*`` ->
    ``"gemini"``. Raises ``ValueError`` for unrecognized prefixes.
    """
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gemini", "gemma")):
        return "gemini"
    raise ValueError(
        f"Cannot detect provider from model {model!r}. "
        "Use a 'claude-*' or 'gemini-*' identifier."
    )


def resolve_api_key(api_key: Optional[str], provider: str) -> str:
    """Return ``api_key`` if non-empty, else look up the provider env var."""
    if api_key:
        return api_key
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider {provider!r}")
    val = os.environ.get(env_var, "").strip()
    if not val:
        raise ValueError(
            f"No API key for {provider}. Pass api_key= or set {env_var}."
        )
    return val


def call_anthropic(
    api_key: str,
    message: str,
    model: str,
    max_tokens: int,
    *,
    timeout: int = DEFAULT_ANTHROPIC_TIMEOUT,
) -> str:
    """Single-turn Claude completion. Returns concatenated text blocks."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "call_anthropic() requires the 'anthropic' SDK. "
            "Install with: pip install metaharmonizer[llm-anthropic]"
        ) from e

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    resp = client.messages.create(
        model=model,
        max_tokens=int(max_tokens),
        messages=[{"role": "user", "content": message}],
    )
    return "\n".join(
        block.text for block in resp.content
        if getattr(block, "type", None) == "text"
    )


def call_gemini(
    api_key: str,
    message: str,
    model: str,
    max_tokens: int,
    *,
    temperature: float = DEFAULT_GEMINI_TEMPERATURE,
    timeout: int = DEFAULT_GEMINI_TIMEOUT,  # noqa: ARG001 — SDK has no per-call timeout
) -> str:
    """Single-turn Gemini completion. Returns response text."""
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "call_gemini() requires the 'google-generativeai' SDK. "
            "Install with: pip install metaharmonizer[llm-gemini]"
        ) from e

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    resp = gen_model.generate_content(
        message,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=int(max_tokens),
            temperature=float(temperature),
        ),
    )
    return resp.text or ""


def call_llm(
    provider: str,
    api_key: str,
    message: str,
    model: str,
    max_tokens: int,
    **kwargs,
) -> str:
    """Dispatch a single-turn call to the named provider."""
    if provider == "anthropic":
        return call_anthropic(api_key, message, model, max_tokens, **kwargs)
    if provider == "gemini":
        return call_gemini(api_key, message, model, max_tokens, **kwargs)
    raise ValueError(f"Unknown provider {provider!r}")
