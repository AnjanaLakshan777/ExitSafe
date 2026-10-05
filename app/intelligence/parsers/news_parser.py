"""Turn raw collected text into clean plain text."""

import html
import re
from dataclasses import replace

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text):
    """Strip HTML tags, decode entities (``&amp;`` -> ``&``) and collapse whitespace."""
    if not text:
        return ""
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE.sub(" ", text).strip()


def parse_news_item(item):
    """Return a copy of a ``RawItem`` with a clean title and body."""
    return replace(item, title=clean_text(item.title), body=clean_text(item.body))
