"""Download one sample image into images/ so you can try run_depth.py immediately."""

from pathlib import Path

import requests

URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats on a couch
DEST = Path(__file__).parent / "images" / "sample.jpg"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    DEST.write_bytes(r.content)
    print(f"Saved {DEST} ({len(r.content) // 1024} KB)")


if __name__ == "__main__":
    main()
