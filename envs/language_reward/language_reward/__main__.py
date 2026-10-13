"""Download the fastText LID model.

    uv run python -m language_reward [destination]

Destination defaults to FASTTEXT_LID_PATH, or lid.176.bin in the repo root.
"""

import os
import sys
import urllib.request

from language_reward import DEFAULT_MODEL_PATH, MODEL_URL


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    if os.path.exists(path):
        print(f"fastText LID model already present at {path}")
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading {MODEL_URL} -> {path}")
    # Download beside the target and rename, so an interrupted or concurrent fetch
    # never leaves a truncated model that loads with garbage predictions.
    partial = f"{path}.{os.getpid()}.part"
    urllib.request.urlretrieve(MODEL_URL, partial)
    os.replace(partial, path)
    print(f"Done ({os.path.getsize(path) // 1_000_000} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
