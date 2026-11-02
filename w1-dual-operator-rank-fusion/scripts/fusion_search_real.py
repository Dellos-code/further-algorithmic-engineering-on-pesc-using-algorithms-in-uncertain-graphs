"""Attempt public real-network sources and write a reproducible acquisition log."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fusion"


def attempt(label, url):
    started = datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read()
        return {"source": label, "url": url, "utc": started, "status": response.status,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "outcome": "downloaded"}
    except Exception as exc:
        return {"source": label, "url": url, "utc": started, "outcome": "failed",
                "error": f"{type(exc).__name__}: {exc}"}


def main():
    sources = [
        ("networkx_karate", "https://raw.githubusercontent.com/networkx/networkx/main/examples/graph/karate_club_graph.py"),
        ("snap_ca_grqc", "https://snap.stanford.edu/data/ca-GrQc.txt.gz"),
        ("snap_email_enron", "https://snap.stanford.edu/data/email-Enron.txt.gz"),
        ("konect", "https://konect.cc/networks/"),
        ("string", "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"),
        ("network_repository", "https://networkrepository.com/"),
        ("netzschleuder", "https://networks.skewed.de/"),
        ("gnn_benchmark", "https://raw.githubusercontent.com/shchur/gnn-benchmark/master/README.md"),
    ]
    records = [attempt(*item) for item in sources]
    (OUT / "data_search_log.json").write_text(json.dumps(records, indent=2) + "\n")
    (OUT / "data_search_log.md").write_text(
        "# Real-network acquisition attempts\n\n" +
        "\n".join(f"- **{r['source']}** `{r['outcome']}`: {r.get('url')} — "
                  f"{r.get('error', r.get('bytes', ''))}" for r in records) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
