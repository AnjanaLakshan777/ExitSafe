"""Gemini requests grounded in Google Search, returning parsed JSON plus the pages Gemini cited."""

import json
import re

from google import genai
from google.genai import errors, types

from app.intelligence.collectors.http import CollectionError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
TIMEOUT_MS = 60_000
# One retry, only for temporary server-side errors (quota errors would just fail again).
RETRY = types.HttpRetryOptions(attempts=2, initial_delay=2.0,
                               http_status_codes=[500, 502, 503, 504])
MAX_ERROR_CHARS = 300


def extract_json(text):
    """Parse the JSON array/object in a model reply, ignoring code fences and surrounding prose."""
    text = _FENCE.sub("", text or "").strip()
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise CollectionError("Gemini reply contained no JSON")
    start = min(starts)
    end = text.rfind("]" if text[start] == "[" else "}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise CollectionError(f"Gemini reply was not valid JSON: {exc}") from exc


def _cited_urls(response):
    urls = []
    for candidate in response.candidates or []:
        metadata = candidate.grounding_metadata
        for chunk in (metadata.grounding_chunks or []) if metadata else []:
            if chunk.web and chunk.web.uri:
                urls.append(chunk.web.uri)
    return urls


def _failure(exc, settings):
    """A short, key-free description of a failed Gemini call."""
    if isinstance(exc, errors.APIError):
        text = f"HTTP {exc.code} {exc.status or ''}".strip()
        if exc.code == 404:
            text += (f": model {settings.gemini_model!r} isn't available to this API key; "
                     "set GEMINI_MODEL to a supported model")
        elif exc.message:
            text += f": {exc.message}"
    else:
        text = f"{type(exc).__name__}: {exc}"
    if settings.gemini_api_key:
        text = text.replace(settings.gemini_api_key, "[redacted]")
    return text[:MAX_ERROR_CHARS]


def grounded_json(prompt, settings):
    """Ask Gemini (with Google Search) and return ``(parsed_json, cited_urls)``."""
    if not settings.gemini_configured:
        raise CollectionError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=settings.gemini_api_key,
                          http_options=types.HttpOptions(timeout=TIMEOUT_MS, retry_options=RETRY))
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
                # Only the built-in search tool is used; this also silences the SDK's AFC warning.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises many error types; report and move on
        raise CollectionError(f"Gemini request failed ({settings.gemini_model}): "
                              f"{_failure(exc, settings)}") from None
    return extract_json(response.text), _cited_urls(response)
