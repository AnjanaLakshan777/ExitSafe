"""Shared HTTP access for collectors: one identifying User-Agent, timeouts and robots.txt checks."""

import urllib.robotparser
from functools import lru_cache
from urllib.parse import urlsplit

import requests

USER_AGENT = "ExitSafeBot/1.0 (market risk research; respects robots.txt)"
TIMEOUT_SECONDS = 20


class CollectionError(RuntimeError):
    """A source could not be read. The bot reports it and carries on with the other sources."""


@lru_cache(maxsize=64)
def _robots(origin):
    parser = urllib.robotparser.RobotFileParser(f"{origin}/robots.txt")
    try:
        response = requests.get(f"{origin}/robots.txt", headers={"User-Agent": USER_AGENT},
                                timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        return None
    if response.status_code in (401, 403):
        parser.disallow_all = True
    elif response.status_code >= 400:
        parser.allow_all = True       # no robots.txt: crawling is not restricted
    else:
        parser.parse(response.text.splitlines())
    return parser


def allowed_by_robots(url):
    """Whether robots.txt lets this bot fetch ``url``. Unreachable robots.txt counts as "no"."""
    parts = urlsplit(url)
    parser = _robots(f"{parts.scheme}://{parts.netloc}")
    return parser is not None and parser.can_fetch(USER_AGENT, url)


def fetch(url, *, method="GET", check_robots=False, data=None):
    """Fetch ``url`` and return the response; raises ``CollectionError`` on any failure."""
    if check_robots and not allowed_by_robots(url):
        raise CollectionError(f"robots.txt does not allow fetching {url}")
    try:
        response = requests.request(method, url, data=data, headers={"User-Agent": USER_AGENT},
                                    timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise CollectionError(f"{url}: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise CollectionError(f"{url}: HTTP {response.status_code}")
    return response
