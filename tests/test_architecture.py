"""Guard the modular-monolith boundaries.

Allowed dependency direction (left may never import right):

    data  ->  analytics  ->  intelligence  ->  recommendation  ->  scheduler

Market data and financial analytics must stay usable without any
intelligence, recommendation or scheduling code.
"""

import ast
import importlib
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"

FORBIDDEN_IMPORTS = {
    "data": ["app.analytics", "app.intelligence", "app.recommendation", "app.scheduler"],
    "analytics": ["app.intelligence", "app.recommendation", "app.scheduler"],
    "intelligence": ["app.recommendation", "app.scheduler"],
    "recommendation": ["app.scheduler"],
}

# Libraries that would mean live network collection has been implemented.
NETWORK_MODULES = ["requests", "httpx", "aiohttp", "urllib.request", "socket",
                   "selenium", "playwright", "bs4", "scrapy"]


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def violations(package, forbidden):
    found = []
    for path in (APP_DIR / package).rglob("*.py"):
        for module in imported_modules(path):
            if any(module == f or module.startswith(f + ".") for f in forbidden):
                found.append(f"{path.relative_to(APP_DIR)} imports {module}")
    return found


@pytest.mark.parametrize("package", sorted(FORBIDDEN_IMPORTS))
def test_domain_boundaries(package):
    assert violations(package, FORBIDDEN_IMPORTS[package]) == []


def test_no_network_collection_yet():
    assert violations("intelligence", NETWORK_MODULES) == []


@pytest.mark.parametrize("package", [
    "intelligence", "intelligence.collectors", "intelligence.parsers",
    "intelligence.classification", "intelligence.impact", "intelligence.signals",
    "recommendation", "scheduler",
])
def test_new_packages_exist(package):
    assert (APP_DIR / package.replace(".", "/") / "__init__.py").is_file()


def test_every_app_module_imports():
    for path in sorted(APP_DIR.rglob("*.py")):
        module = ".".join(path.relative_to(APP_DIR.parent).with_suffix("").parts)
        importlib.import_module(module)
