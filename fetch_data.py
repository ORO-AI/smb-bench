"""Fetch what recompute.py reads from the public dataset, at the revision this repository's copies come from.

    python fetch_data.py

Writes data/runs.parquet (the one file recompute.py and plot.py read) and dataset-card.md (the dataset's README.md,
for `recompute.py --check`), then confirms that the files this repository copies from the dataset are byte-identical
to the dataset's own at that revision. Standard library only; no account or token is needed.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATASET = "oro-agents/smb-bench"
REVISION = "d7c5b3f4d7940ef4be600b230ab29af3682f3fdf"
HERE = Path(__file__).resolve().parent
FETCHED = {"data/runs.parquet": "data/runs.parquet", "README.md": "dataset-card.md"}  # dataset path -> local path
COPIED = ["recompute.py", "plot.py", "requirements.txt", "task/buyer-prompt-v11.json"]


def get(path):
    """The bytes of one dataset file at the pinned revision."""
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "smb-bench-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    print(f"{DATASET} at {REVISION}")
    for source, dest in FETCHED.items():
        data = get(source)
        out = HERE / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"  fetched  {dest}  {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()}")
    differs = []
    for path in COPIED:
        local = (HERE / path).read_bytes()
        same = local == get(path)
        differs += [] if same else [path]
        print(f"  {'same   ' if same else 'DIFFERS'}  {path}  sha256 {hashlib.sha256(local).hexdigest()}")
    if REVISION not in (HERE / "README.md").read_text():
        differs.append("README.md (it names another revision)")
    if differs:
        print(f"{len(differs)} file(s) differ from the dataset at this revision: {', '.join(differs)}")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
