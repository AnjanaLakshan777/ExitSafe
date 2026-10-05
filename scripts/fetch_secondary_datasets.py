"""Download the secondary (research) datasets at pinned revisions.

    python scripts/fetch_secondary_datasets.py

Files go to data/raw/external/huggingface/<owner>__<name>/ (not tracked in
git) together with a SOURCE.json recording the URL, revision, licence,
retrieval time and SHA-256 of every file, which the audit script uses for
provenance. These are plain downloads of published dataset files; nothing is
scraped.
"""

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.paths import EXTERNAL_RAW_DIR  # noqa: E402

HF_DIR = EXTERNAL_RAW_DIR / "huggingface"

# Pinned revisions: a new upstream version must be adopted deliberately.
DATASETS = [
    {
        "source_name": "hf_tharu_jwd_cse_market_data",
        "repo": "tharu-jwd/cse-market-data",
        "revision": "669594502b85772c7b7c853067afde91402e3c2a",
        "license": "CC-BY-4.0",
        "files": ["README.md", "stock_prices.csv", "aspi.csv", "sector_mapping.csv"],
    },
    {
        "source_name": "hf_kjhq_sri_lanka_stock_symbols",
        "repo": "kjhq/Sri-Lanka-Stock-Symbols-and-Metadata",
        "revision": "37c8ea035c335f8c55daba0f215087f37fe11f34",
        "license": "CC0-1.0",
        "files": ["README.md", "srilanka.csv"],
    },
]


def dataset_dir(repo):
    return HF_DIR / repo.replace("/", "__")


def fetch(dataset):
    target = dataset_dir(dataset["repo"])
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in dataset["files"]:
        url = (f"https://huggingface.co/datasets/{dataset['repo']}/resolve/"
               f"{dataset['revision']}/{name}")
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        (target / name).write_bytes(content)
        files[name] = {"url": url, "sha256": hashlib.sha256(content).hexdigest(),
                       "size_bytes": len(content)}
        print(f"  {name}: {len(content):,} bytes")

    record = {
        "source_name": dataset["source_name"],
        "repo": dataset["repo"],
        "revision": dataset["revision"],
        "license": dataset["license"],
        "dataset_url": f"https://huggingface.co/datasets/{dataset['repo']}",
        "retrieval_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    (target / "SOURCE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    for dataset in DATASETS:
        print(f"{dataset['repo']} @ {dataset['revision'][:10]}")
        fetch(dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
