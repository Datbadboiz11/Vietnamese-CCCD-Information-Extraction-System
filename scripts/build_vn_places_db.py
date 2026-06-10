"""Build compact Vietnamese administrative division lookup from raw data.

Reads the raw JSON files downloaded from github.com/madnh/hanhchinhvn
and produces a single compact JSON optimized for OCR fuzzy matching.

Usage:
    python scripts/build_vn_places_db.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "vn_administrative"
OUTPUT = RAW_DIR / "divisions_lookup.json"


def main() -> None:
    tinh = RAW_DIR / "tinh_tp.json"
    huyen = RAW_DIR / "quan_huyen.json"
    xa = RAW_DIR / "xa_phuong.json"

    for f in (tinh, huyen, xa):
        if not f.exists():
            print(f"Missing: {f}", file=sys.stderr)
            sys.exit(1)

    with open(tinh, encoding="utf-8") as f:
        provinces_raw = json.load(f)
    with open(huyen, encoding="utf-8") as f:
        districts_raw = json.load(f)
    with open(xa, encoding="utf-8") as f:
        wards_raw = json.load(f)

    provinces: dict[str, list[str]] = {}
    for info in provinces_raw.values():
        provinces[info["slug"]] = [info["name"], info["code"]]

    districts: dict[str, dict[str, list[str]]] = {}
    for info in districts_raw.values():
        pc = info["parent_code"]
        districts.setdefault(pc, {})[info["slug"]] = [info["name_with_type"], info["code"]]

    wards: dict[str, dict[str, str]] = {}
    for info in wards_raw.values():
        dc = info["parent_code"]
        wards.setdefault(dc, {})[info["slug"]] = info["name_with_type"]

    lookup = {"p": provinces, "d": districts, "w": wards}

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = OUTPUT.stat().st_size / 1024
    print(
        f"Built {OUTPUT.name}: "
        f"{len(provinces)} provinces, "
        f"{sum(len(v) for v in districts.values())} districts, "
        f"{sum(len(v) for v in wards.values())} wards "
        f"({size_kb:.0f} KB)"
    )


if __name__ == "__main__":
    main()
