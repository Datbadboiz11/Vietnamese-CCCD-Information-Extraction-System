"""
Prepare VietOCR fine-tuning data from reviewed + pseudo-labels.

v2 improvements:
  - Higher confidence thresholds for address/origin pseudo-labels
  - Field balancing: upsample underrepresented hard fields
  - Include high-confidence "review" bucket samples for address/origin
  - Better text validation

Priority order:
  1. ground_truth_text from reviewed.jsonl (manually corrected)
  2. best_text from reviewed.jsonl where review_bucket == "accept"
  3. High-confidence "review" bucket (address/origin only, conf >= 0.88)
  4. best_text from pseudo_labels.jsonl (confidence-filtered)

Records rejected in reviewed.jsonl are excluded even if they pass
confidence filtering from pseudo_labels.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWED = PROJECT_ROOT / "data" / "processed" / "ocr" / "reviewed.jsonl"
PSEUDO_LABELS = PROJECT_ROOT / "data" / "processed" / "ocr" / "pseudo_labels.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ocr" / "finetune"

OLD_PREFIX = "data/interim/cropped_fields/"
NEW_PREFIX = "data/processed/ocr/field_crops/"

CONF_THRESHOLDS = {
    "id": 0.90,
    "birth": 0.90,
    "name": 0.82,
    "address": 0.72,
    "origin": 0.72,
}

REVIEW_BUCKET_CONF_THRESHOLDS = {
    "address": 0.88,
    "origin": 0.88,
}

GT_UPWEIGHT_FACTOR = 4
HARD_FIELD_UPSAMPLE = {
    "address": 2,
    "origin": 2,
}
TARGET_BALANCE_MAX = 6000

VAL_RATIO = 0.1
SEED = 42

_LABEL_RE = re.compile(
    r"(qu[aáeêế]\s*qu[aáâ]n\s*:?\s*|"
    r"n[oơ]i?\s*(?:d?k?h?k?\s*)?th[uư]?r?[oơờ]?ng\s*tr[uú]?\s*:?\s*|"
    r"nguy[eê]n\s*qu[aá]n\s*:?\s*|"
    r"place\s*[ao][fl]\s*\w+\s*:?\s*|"
    r"p[ilh]?[aei]c[eo]?\s*[dao]?f?\s*\w*\s*:?\s*|"
    r"\borigin\s*:?\s*|\bresidence?\s*:?\s*)",
    re.IGNORECASE,
)


def remap_crop_path(old_path: str) -> str:
    if old_path.startswith(OLD_PREFIX):
        return old_path.replace(OLD_PREFIX, NEW_PREFIX, 1)
    return old_path


_OOV_FIX = str.maketrans("āīūåüÖ", "ãĩũắũÕ")


def fix_oov_chars(text: str) -> str:
    return text.translate(_OOV_FIX)


def clean_label_contamination(text: str, field: str) -> str:
    if field not in ("address", "origin"):
        return text
    cleaned = _LABEL_RE.sub("", text).strip().lstrip(":/-.,; ")
    return cleaned if cleaned else text


def is_valid_text(text: str, field: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False

    digits = sum(c.isdigit() for c in text)

    if field == "id":
        return bool(re.match(r"^\d{9,12}$", text.strip()))
    if field == "birth":
        return bool(re.match(r"^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$", text.strip()))

    if field in ("name", "address", "origin"):
        if digits > len(text) * 0.5 and digits >= 6:
            return False
        has_viet = bool(re.search(
            r"[ăâđêôơưàáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]",
            text, re.IGNORECASE,
        ))
        if re.match(r"^[A-Z][a-z]+", text) and not has_viet and len(text) > 5:
            return False
        words = text.split()
        if words and len(text) > 30 and any(text.count(w) > 5 for w in words[:3] if len(w) >= 2):
            return False
        if field in ("address", "origin"):
            alpha_count = sum(c.isalpha() for c in text)
            if alpha_count < 3:
                return False
            if len(text) < 5:
                return False

    return True


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Load reviewed.jsonl
    # ------------------------------------------------------------------
    reviewed_records: list[dict] = []
    if REVIEWED.exists():
        with open(REVIEWED, encoding="utf-8") as f:
            reviewed_records = [json.loads(line) for line in f if line.strip()]
    print(f"Reviewed records: {len(reviewed_records)}")

    reviewed_by_path: dict[str, dict] = {}
    rejected_paths: set[str] = set()

    for row in reviewed_records:
        crop_path = remap_crop_path(row.get("crop_path", ""))
        bucket = row.get("review_bucket", "")
        reviewed_by_path[crop_path] = row
        if bucket == "reject":
            rejected_paths.add(crop_path)

    # ------------------------------------------------------------------
    # Phase 2: Collect training samples from reviewed data
    # ------------------------------------------------------------------
    accepted: dict[str, list[tuple[str, str, str]]] = {f: [] for f in CONF_THRESHOLDS}
    used_paths: set[str] = set()
    stats = Counter()

    for row in reviewed_records:
        field = row.get("class", "")
        if field not in CONF_THRESHOLDS:
            continue

        bucket = row.get("review_bucket", "")
        crop_path = remap_crop_path(row.get("crop_path", ""))
        abs_path = PROJECT_ROOT / crop_path

        if not abs_path.exists():
            stats[f"{field}_missing_file"] += 1
            continue

        # Priority 1: manually corrected ground truth
        gt = fix_oov_chars((row.get("ground_truth_text") or "").strip())
        if gt and is_valid_text(gt, field):
            gt = clean_label_contamination(gt, field)
            if gt and is_valid_text(gt, field):
                accepted[field].append((crop_path, gt, "reviewed_gt"))
                used_paths.add(crop_path)
                stats[f"{field}_reviewed_gt"] += 1
                continue

        # Priority 2: accepted by reviewer
        if bucket == "accept":
            text = fix_oov_chars((row.get("best_text") or "").strip())
            text = clean_label_contamination(text, field)
            if text and is_valid_text(text, field):
                accepted[field].append((crop_path, text, "reviewed_accept"))
                used_paths.add(crop_path)
                stats[f"{field}_reviewed_accept"] += 1
                continue
            stats[f"{field}_accept_invalid"] += 1

        # Priority 3: high-confidence "review" bucket for hard fields
        if bucket == "review" and field in REVIEW_BUCKET_CONF_THRESHOLDS:
            conf = row.get("best_conf", 0)
            if conf >= REVIEW_BUCKET_CONF_THRESHOLDS[field]:
                text = fix_oov_chars((row.get("best_text") or "").strip())
                text = clean_label_contamination(text, field)
                if text and is_valid_text(text, field):
                    accepted[field].append((crop_path, text, "review_high_conf"))
                    used_paths.add(crop_path)
                    stats[f"{field}_review_high_conf"] += 1
                    continue

        if bucket == "reject":
            stats[f"{field}_reviewed_reject"] += 1
        if bucket == "review":
            stats[f"{field}_reviewed_pending"] += 1

    print("\nPhase 1 — Reviewed data:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")

    reviewed_crop_paths = {
        path for path, row in reviewed_by_path.items()
        if row.get("review_bucket") in ("accept", "reject")
    }

    # ------------------------------------------------------------------
    # Phase 3: Fill remaining from pseudo_labels.jsonl
    # ------------------------------------------------------------------
    pseudo_stats = Counter()
    if PSEUDO_LABELS.exists():
        with open(PSEUDO_LABELS, encoding="utf-8") as f:
            pseudo_rows = [json.loads(line) for line in f if line.strip()]
        print(f"\nPseudo-labels: {len(pseudo_rows)}")

        for row in pseudo_rows:
            field = row.get("class", "")
            if field not in CONF_THRESHOLDS:
                continue

            crop_path = remap_crop_path(row.get("crop_path", ""))

            if crop_path in used_paths or crop_path in reviewed_crop_paths:
                pseudo_stats[f"{field}_already_used"] += 1
                continue

            if crop_path in rejected_paths:
                pseudo_stats[f"{field}_rejected"] += 1
                continue

            text = (row.get("best_text") or "").strip()
            conf = row.get("best_conf", 0)
            threshold = CONF_THRESHOLDS[field]

            if conf < threshold:
                pseudo_stats[f"{field}_low_conf"] += 1
                continue

            abs_path = PROJECT_ROOT / crop_path
            if not abs_path.exists():
                pseudo_stats[f"{field}_missing_file"] += 1
                continue

            text = clean_label_contamination(text, field)
            if not is_valid_text(text, field):
                pseudo_stats[f"{field}_invalid_text"] += 1
                continue

            accepted[field].append((crop_path, text, "pseudo_label"))
            used_paths.add(crop_path)
            pseudo_stats[f"{field}_pseudo_added"] += 1

        print("\nPhase 2 — Pseudo-label fill:")
        for key in sorted(pseudo_stats):
            print(f"  {key}: {pseudo_stats[key]}")

    # ------------------------------------------------------------------
    # Phase 4: Summary, balance, & split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Before balancing:")
    for field in sorted(accepted):
        samples = accepted[field]
        sources = Counter(src for _, _, src in samples)
        print(f"  {field}: {len(samples)} — {dict(sources)}")

    all_train = []
    all_val = []

    for field, samples in accepted.items():
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * VAL_RATIO))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]

        # GT upweighting
        upweighted_train = []
        gt_count = 0
        for item in train_samples:
            upweighted_train.append(item)
            if item[2] == "reviewed_gt":
                for _ in range(GT_UPWEIGHT_FACTOR - 1):
                    upweighted_train.append(item)
                gt_count += 1

        # Hard field upsampling
        upsample_factor = HARD_FIELD_UPSAMPLE.get(field, 1)
        if upsample_factor > 1:
            base = list(upweighted_train)
            for _ in range(upsample_factor - 1):
                upweighted_train.extend(base)

        # Cap at target balance max
        if len(upweighted_train) > TARGET_BALANCE_MAX:
            random.shuffle(upweighted_train)
            upweighted_train = upweighted_train[:TARGET_BALANCE_MAX]

        all_train.extend(upweighted_train)
        all_val.extend(val_samples)

        print(f"  {field}: train={len(upweighted_train)} (base={len(train_samples)}, "
              f"GT upweight={gt_count}x{GT_UPWEIGHT_FACTOR}, "
              f"field_upsample={upsample_factor}x), val={len(val_samples)}")

    random.shuffle(all_train)
    random.shuffle(all_val)

    total_train = len(all_train)
    total_val = len(all_val)

    train_path = OUTPUT_DIR / "train_annotation.txt"
    val_path = OUTPUT_DIR / "val_annotation.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        for crop_path, text, _src in all_train:
            f.write(f"{crop_path}\t{text}\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for crop_path, text, _src in all_val:
            f.write(f"{crop_path}\t{text}\n")

    print(f"\nWrote {total_train} train → {train_path}")
    print(f"Wrote {total_val} val   → {val_path}")

    # Field distribution in final train set
    print("\nFinal train field distribution:")
    field_counts = Counter()
    for crop_path, text, _src in all_train:
        m = re.search(r"field_crops/(\w+)/", crop_path)
        if m:
            field_counts[m.group(1)] += 1
    for field, count in field_counts.most_common():
        pct = count / max(1, total_train) * 100
        print(f"  {field:10s}: {count:6d} ({pct:.1f}%)")

    print("\nSample annotations (train):")
    for crop_path, text, src in all_train[:10]:
        print(f"  [{src:16s}] {crop_path}\t{text}")


if __name__ == "__main__":
    main()
