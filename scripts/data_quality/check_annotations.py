"""
Bước 3: Kiểm tra chất lượng annotation

Kiểm tra:
  1. Ảnh thiếu annotation (0 bbox)
  2. Ảnh thiếu class "card"
  3. Ảnh có card nhưng thiếu field (< 4 field classes)
  4. Bbox quá nhỏ (area < 100 px²)
  5. Bbox quá lớn (> 95% diện tích ảnh)
  6. Bbox nằm ngoài ảnh
  7. Bbox overlap bất thường cùng class (IoU > 0.9)
  8. Phân bố class

Output:
  data/interim/annotation_quality_report.md
  data/interim/flagged_images.json

Chạy từ thư mục gốc project:
  python scripts/data_quality/check_annotations.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path("data/cccd.v1i.coco")
INTERIM_DIR = Path("data/interim")
SPLITS = ["train", "valid", "test"]

FIELD_CLASSES = {"id", "name", "birth", "origin", "address", "title"}
MIN_BBOX_AREA = 100       # px²
MAX_BBOX_RATIO = 0.95     # % diện tích ảnh
MIN_FIELD_COUNT = 4       # số field tối thiểu nếu có card
IOU_THRESHOLD = 0.9



def bbox_area(bbox):
    """COCO bbox: [x, y, w, h]"""
    return bbox[2] * bbox[3]


def compute_iou(b1, b2):
    """IoU của 2 COCO bbox [x, y, w, h]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = bbox_area(b1) + bbox_area(b2) - inter
    return inter / union if union > 0 else 0.0



def check_split(split: str) -> tuple[list[dict], Counter, int]:
    """Trả về (flagged_images, class_dist, total_images)."""
    ann_path = DATA_DIR / split / "_annotations.coco.json"
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # Nhóm annotation theo image
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    class_dist: Counter = Counter()
    flagged: list[dict] = []

    for img in tqdm(coco["images"], desc=f"  {split}", ncols=80, leave=False):
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        img_area = img_w * img_h
        anns = anns_by_image.get(img_id, [])

        issues: list[str] = []
        severity: str = "warning"

        # Đếm class distribution
        classes_present = [cat_id_to_name.get(a["category_id"], "?") for a in anns]
        for cls in classes_present:
            class_dist[cls] += 1

        # 1. Thiếu annotation
        if len(anns) == 0:
            issues.append("no_annotation")

        # 2. Thiếu class card
        if "card" not in classes_present:
            issues.append("missing_card_class")

        # 3. Có card nhưng thiếu field
        field_count = sum(1 for c in classes_present if c in FIELD_CLASSES)
        if "card" in classes_present and field_count < MIN_FIELD_COUNT:
            issues.append(f"few_fields:{field_count}")

        # 4–6. Kiểm tra từng bbox
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, w, h]
            area = bbox_area(bbox)
            cls = cat_id_to_name.get(ann["category_id"], "?")

            # 4. Bbox quá nhỏ
            if area < MIN_BBOX_AREA:
                issues.append(f"bbox_too_small:{cls}:{area:.0f}px2")
                severity = "error"

            # 5. Bbox quá lớn
            if area > img_area * MAX_BBOX_RATIO:
                issues.append(f"bbox_too_large:{cls}:{area / img_area * 100:.0f}%")
                severity = "error"

            # 6. Bbox ngoài ảnh
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                issues.append(f"bbox_out_of_bounds:{cls}")
                severity = "error"

        # 7. Overlap bất thường cùng class
        by_class: dict[str, list] = defaultdict(list)
        for ann in anns:
            cls = cat_id_to_name.get(ann["category_id"], "?")
            by_class[cls].append(ann["bbox"])

        for cls, bboxes in by_class.items():
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    iou = compute_iou(bboxes[i], bboxes[j])
                    if iou > IOU_THRESHOLD:
                        issues.append(f"high_overlap:{cls}:IoU={iou:.2f}")

        if issues:
            flagged.append({
                "split": split,
                "image_id": img_id,
                "file_name": img["file_name"],
                "severity": severity,
                "issues": issues,
            })

    return flagged, class_dist, len(coco["images"])


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    all_flagged: list[dict] = []
    all_class_dist: Counter = Counter()
    total_images = 0

    for split in SPLITS:
        flagged, class_dist, n_imgs = check_split(split)
        all_flagged.extend(flagged)
        all_class_dist.update(class_dist)
        total_images += n_imgs

    # Lưu flagged images
    out_flagged = INTERIM_DIR / "flagged_images.json"
    with open(out_flagged, "w", encoding="utf-8") as f:
        json.dump(all_flagged, f, ensure_ascii=False, indent=2)

    # Phân loại issues
    error_images = [f for f in all_flagged if f["severity"] == "error"]
    warning_images = [f for f in all_flagged if f["severity"] == "warning"]
    error_rate = len(all_flagged) / total_images * 100
    pass_criteria = error_rate < 5.0

    # Đếm loại issue
    issue_counter: Counter = Counter()
    for f in all_flagged:
        for issue in f["issues"]:
            issue_type = issue.split(":")[0]
            issue_counter[issue_type] += 1

    # Báo cáo
    lines = [
        "# Annotation Quality Report\n",
        "## Tong quan",
        f"- Tong so anh: {total_images}",
        f"- Anh co van de: {len(all_flagged)} ({error_rate:.1f}%)",
        f"  - ERROR: {len(error_images)}",
        f"  - WARNING: {len(warning_images)}",
        f"- Tieu chi pass (< 5% loi): {'PASS' if pass_criteria else 'FAIL'}",
        "",
        "## Phan bo class",
    ]
    for cls, count in sorted(all_class_dist.items()):
        status = "OK" if count >= 100 else "FAIL (<100)"
        lines.append(f"- {cls}: {count} annotations  [{status}]")

    lines += [
        "",
        "## Phan loai issues",
    ]
    for issue_type, count in issue_counter.most_common():
        lines.append(f"- {issue_type}: {count} anh")

    lines += [
        "",
        "## Ket luan",
        f"- {'PASS' if pass_criteria else 'FAIL'}: {error_rate:.1f}% anh co loi (nguong < 5%)",
        f"- Tat ca class co >= 100 annotation: {'YES' if all(c >= 100 for c in all_class_dist.values()) else 'NO'}",
    ]

    out_report = INTERIM_DIR / "annotation_quality_report.md"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Tong anh: {total_images}")
    print(f"  Anh co van de: {len(all_flagged)} ({error_rate:.1f}%)")
    print(f"  ERROR: {len(error_images)}  |  WARNING: {len(warning_images)}")
    print(f"  Tieu chi pass: {'PASS' if pass_criteria else 'FAIL'}")
    print(f"  Phan bo class: {dict(all_class_dist)}")
    print(f"  -> {out_flagged}")
    print(f"  -> {out_report}")


if __name__ == "__main__":
    main()
