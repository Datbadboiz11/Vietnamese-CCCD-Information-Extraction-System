"""
Tạo ocr_performance_report.json từ pseudo_labels.jsonl và reviewed.jsonl.

Output file dùng bởi demo/app.py để hiển thị phần "OCR Performance Metrics".

Chạy:
    python scripts/ocr_performance_report.py
    # hoặc chỉ định đường dẫn
    python scripts/ocr_performance_report.py \
        --pseudo  pseudo_labels.jsonl \
        --reviewed reviewed.jsonl \
        --output  ocr_performance_report.json

Cấu trúc output:
    {
        "confidence_analysis": {
            "<class>": {
                "total_samples": int,
                "overall": {
                    "mean_conf": float,
                    "median_conf": float,
                    "high_conf_pct": float,  # % rows >= 0.8
                    "low_conf_pct": float,   # % rows < 0.5
                }
            }
        },
        "quality_analysis": {
            "<class>": {
                "overall": {
                    "samples": int,
                    "accuracy": float,        # exact-match rate
                    "correct_samples": int,
                    "error_samples": int,
                    "cer": float,
                    "wer": float,
                }
            }
        },
        "generated_at": str,
        "pseudo_labels_path": str,
        "reviewed_path": str,
    }
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "").strip()
    return " ".join(value.split())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / max(1, len(ref))


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / max(1, len(r))


# ---------------------------------------------------------------------------
# Confidence analysis (from pseudo_labels.jsonl)
# ---------------------------------------------------------------------------

def compute_confidence_analysis(rows: list[dict]) -> dict:
    """
    Per class: mean/median confidence, % high (>=0.8), % low (<0.5).
    Uses 'best_conf' field from pseudo_labels.
    """
    per_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cls = row.get("class") or row.get("field_name") or ""
        conf = row.get("best_conf")
        if cls and conf is not None:
            try:
                per_class[cls].append(float(conf))
            except (TypeError, ValueError):
                pass

    result = {}
    for cls, confs in sorted(per_class.items()):
        n = len(confs)
        if n == 0:
            continue
        mean_c = sum(confs) / n
        median_c = statistics.median(confs)
        high_pct = round(sum(1 for c in confs if c >= 0.8) / n * 100, 2)
        low_pct = round(sum(1 for c in confs if c < 0.5) / n * 100, 2)
        result[cls] = {
            "total_samples": n,
            "overall": {
                "mean_conf": round(mean_c, 4),
                "median_conf": round(median_c, 4),
                "high_conf_pct": high_pct,
                "low_conf_pct": low_pct,
            },
        }
    return result


# ---------------------------------------------------------------------------
# Quality analysis (from reviewed.jsonl — has ground_truth_text)
# ---------------------------------------------------------------------------

def compute_quality_analysis(rows: list[dict]) -> dict:
    """
    Per class: exact match accuracy, CER, WER.
    Uses 'best_text' vs 'ground_truth_text' from reviewed.jsonl.
    """
    per_class: dict[str, dict] = defaultdict(lambda: {
        "samples": 0, "correct": 0, "cer_total": 0.0, "wer_total": 0.0
    })

    for row in rows:
        cls = row.get("class") or row.get("field_name") or ""
        gt_raw = row.get("ground_truth_text") or row.get("text") or ""
        pred_raw = row.get("best_text") or row.get("predicted_text") or ""
        gt = _normalize(gt_raw)
        pred = _normalize(pred_raw)

        if not gt:
            continue

        bucket = per_class[cls]
        bucket["samples"] += 1
        if gt == pred:
            bucket["correct"] += 1
        bucket["cer_total"] += _cer(gt, pred)
        bucket["wer_total"] += _wer(gt, pred)

    result = {}
    for cls, d in sorted(per_class.items()):
        n = d["samples"]
        if n == 0:
            continue
        correct = d["correct"]
        result[cls] = {
            "overall": {
                "samples": n,
                "accuracy": round(correct / n, 4),
                "correct_samples": correct,
                "error_samples": n - correct,
                "cer": round(d["cer_total"] / n, 4),
                "wer": round(d["wer_total"] / n, 4),
            }
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(
    pseudo_path: str,
    reviewed_path: str,
    output_path: str,
) -> dict:
    print(f"Loading pseudo_labels: {pseudo_path}")
    pseudo_rows = _load_jsonl(pseudo_path)
    print(f"  → {len(pseudo_rows)} rows")

    print(f"Loading reviewed: {reviewed_path}")
    reviewed_rows = _load_jsonl(reviewed_path)
    print(f"  → {len(reviewed_rows)} rows")

    print("Computing confidence analysis...")
    conf_analysis = compute_confidence_analysis(pseudo_rows)

    print("Computing quality analysis...")
    quality_analysis = compute_quality_analysis(reviewed_rows)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pseudo_labels_path": str(pseudo_path),
        "reviewed_path": str(reviewed_path),
        "pseudo_labels_count": len(pseudo_rows),
        "reviewed_count": len(reviewed_rows),
        "confidence_analysis": conf_analysis,
        "quality_analysis": quality_analysis,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"Report saved → {out}")

    # Print quick summary
    print("\n=== Confidence Summary ===")
    for cls, data in conf_analysis.items():
        ov = data["overall"]
        print(f"  {cls:10} | n={data['total_samples']:5} | "
              f"mean={ov['mean_conf']:.3f} | high%={ov['high_conf_pct']:5.1f}% | "
              f"low%={ov['low_conf_pct']:4.1f}%")

    print("\n=== Quality Summary (vs ground truth) ===")
    for cls, data in quality_analysis.items():
        ov = data["overall"]
        print(f"  {cls:10} | n={ov['samples']:5} | "
              f"accuracy={ov['accuracy']:.3f} | "
              f"CER={ov['cer']:.3f} | WER={ov['wer']:.3f}")

    return report


def main() -> None:
    # Default paths: look relative to project root
    project_root = Path(__file__).parent.parent
    default_pseudo = project_root / "pseudo_labels.jsonl"
    default_reviewed = project_root / "reviewed.jsonl"
    default_output = project_root / "ocr_performance_report.json"

    parser = argparse.ArgumentParser(description="Generate OCR performance report")
    parser.add_argument("--pseudo", default=str(default_pseudo),
                        help="Path to pseudo_labels.jsonl")
    parser.add_argument("--reviewed", default=str(default_reviewed),
                        help="Path to reviewed.jsonl (with ground_truth_text)")
    parser.add_argument("--output", default=str(default_output),
                        help="Output path for ocr_performance_report.json")
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.pseudo).exists():
        raise FileNotFoundError(f"pseudo_labels file not found: {args.pseudo}")
    if not Path(args.reviewed).exists():
        raise FileNotFoundError(f"reviewed file not found: {args.reviewed}")

    generate_report(args.pseudo, args.reviewed, args.output)


if __name__ == "__main__":
    main()
