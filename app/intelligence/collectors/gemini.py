"""Gemini requests grounded in Google Search, returning parsed JSON plus the pages Gemini cited."""

import json
import re

from google import genai
from google.genai import types

from app.intelligence.collectors.http import CollectionError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


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


def grounded_json(prompt, settings):
    """Ask Gemini (with Google Search) and return ``(parsed_json, cited_urls)``."""
    if not settings.gemini_configured:
        raise CollectionError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises many error types; report and move on
        raise CollectionError(f"Gemini request failed ({settings.gemini_model}): {exc}") from exc
    return extract_json(response.text), _cited_urls(response)
