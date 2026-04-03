"""
Bước 2: Kiểm tra ảnh trùng lặp (Deduplication)

Thuật toán:
  - Tính pHash cho mỗi ảnh (DCT-based perceptual hash)
  - Cluster theo Hamming distance <= 8 (Union-Find)
  - Ngoài ra group theo base name (strip Roboflow rf. suffix) để bắt leakage
    từ augmented copies — pHash threshold=8 không đủ bắt augmented versions

Output:
  data/interim/dedup_clusters.json
  data/interim/dedup_report.md

Chạy từ thư mục gốc project:
  python scripts/data_quality/dedup_images.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from tqdm import tqdm

DATA_DIR = Path("data/cccd.v1i.coco")
INTERIM_DIR = Path("data/interim")
SPLITS = ["train", "valid", "test"]
HAMMING_THRESHOLD = 8
HASH_SIZE = 8  



def compute_phash(image_path: Path, hash_size: int = HASH_SIZE) -> np.ndarray | None:
    """Perceptual hash: DCT trên ảnh grayscale 32x32, lấy 8x8 top-left."""
    img_size = hash_size * 4
    img = Image.open(image_path).convert("L").resize(
        (img_size, img_size), Image.LANCZOS
    )
    pixels = np.array(img, dtype=float)
    dct_coeffs = dct(dct(pixels, axis=0), axis=1)
    dct_low = dct_coeffs[:hash_size, :hash_size]
    median = np.median(dct_low)
    return (dct_low > median).flatten()

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

    def clusters(self) -> list[list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return list(groups.values())


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load tất cả ảnh từ 3 split
    all_images: list[dict] = []
    for split in SPLITS:
        ann_path = DATA_DIR / split / "_annotations.coco.json"
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)
        for img in coco["images"]:
            all_images.append({
                "split": split,
                "image_id": img["id"],
                "file_name": img["file_name"],
                "full_path": str(DATA_DIR / split / img["file_name"]),
                "base_name": re.sub(r"_jpg\.rf\.[a-zA-Z0-9]+\.jpg$", "", img["file_name"]),
            })

    n = len(all_images)
    print(f"Tong so anh: {n}")

    # 2. Tính pHash
    print("Tinh pHash:")
    hashes: list[np.ndarray | None] = []
    failed: list[str] = []
    for item in tqdm(all_images, ncols=80):
        h = compute_phash(Path(item["full_path"]))
        hashes.append(h)
        if h is None:
            failed.append(item["file_name"])

    # 3. pHash clustering (Hamming distance <= threshold)
    print("Cap trung lap:")
    valid_idx = [i for i, h in enumerate(hashes) if h is not None]
    hash_matrix = np.array([hashes[i] for i in valid_idx])  # (m, 64)

    uf_phash = UnionFind(n)
    phash_pairs = 0

    for pos, i in enumerate(tqdm(valid_idx, ncols=80)):
        if pos + 1 >= len(valid_idx):
            break
        remaining = hash_matrix[pos + 1:]
        distances = np.sum(hash_matrix[pos] != remaining, axis=1)
        matches = np.where(distances <= HAMMING_THRESHOLD)[0]
        for m in matches:
            j = valid_idx[pos + 1 + m]
            uf_phash.union(i, j)
            phash_pairs += 1

    phash_clusters = uf_phash.clusters()
    phash_dup_clusters = [c for c in phash_clusters if len(c) > 1]
    print(f"  Cap trung: {phash_pairs} | Cluster >= 2 anh: {len(phash_dup_clusters)}")

    # 4. Base-name clustering (bắt Roboflow augmented copies)
    print("Nhóm theo base name:")
    base_groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(all_images):
        base_groups[item["base_name"]].append(i)

    basename_dup = {k: v for k, v in base_groups.items() if len(v) > 1}
    print(f"  Base names unique: {len(base_groups)}")
    print(f"  Base names co >= 2 ban augment: {len(basename_dup)}")

    # 5. Cluster cuối để split = base-name grouping (primary)
    #    pHash chỉ dùng để báo cáo, KHÔNG merge cross base-name
    #    Lý do: CCCD có template cố định → pHash với Union-Find gây chaining
    #    (image A~B, B~C → A,B,C cùng cluster dù A và C không liên quan)
    final_clusters = list(base_groups.values())
    final_dup = [c for c in final_clusters if len(c) > 1]

    # 6. Kiểm tra cross-split leakage
    leakage_count = 0
    for cluster in final_clusters:
        splits_in = set(all_images[i]["split"] for i in cluster)
        if len(splits_in) > 1:
            leakage_count += 1

    # 7. Lưu clusters
    output_clusters = []
    for cluster_id, indices in enumerate(
        sorted(final_clusters, key=len, reverse=True)
    ):
        output_clusters.append({
            "cluster_id": cluster_id,
            "size": len(indices),
            "images": [
                {
                    "split": all_images[i]["split"],
                    "file_name": all_images[i]["file_name"],
                    "full_path": all_images[i]["full_path"],
                    "base_name": all_images[i]["base_name"],
                }
                for i in indices
            ],
        })

    out_clusters = INTERIM_DIR / "dedup_clusters.json"
    with open(out_clusters, "w", encoding="utf-8") as f:
        json.dump(output_clusters, f, ensure_ascii=False, indent=2)

    # 8. Báo cáo
    num_dup_images = sum(c["size"] for c in output_clusters if c["size"] > 1)
    size_dist = Counter(c["size"] for c in output_clusters)

    report_lines = [
        "# Deduplication Report\n",
        "## Tong quan",
        f"- Tong so anh: {n}",
        f"- So anh bi loi hash: {len(failed)}",
        f"- Tong so cluster: {len(final_clusters)}",
        f"  - Cluster >= 2 anh (co ban trung/augment): {len(final_dup)}",
        f"  - Cluster don le (unique): {len(final_clusters) - len(final_dup)}",
        f"- Anh nam trong cluster trung: {num_dup_images} ({num_dup_images / n * 100:.1f}%)",
        "",
        "## pHash (Hamming <= 8)",
        f"- Cap trung: {phash_pairs}",
        f"- Cluster pHash >= 2: {len(phash_dup_clusters)}",
        "",
        "## Base-name grouping (Roboflow augmented copies)",
        f"- So base name unique (anh goc): {len(base_groups)}",
        f"- Base name co >= 2 ban: {len(basename_dup)}",
        "",
        "## Phan bo kich thuoc cluster",
    ]
    for size in sorted(size_dist.keys()):
        report_lines.append(f"- Cluster size {size}: {size_dist[size]} cluster")

    report_lines += [
        "",
        "## Cross-split leakage (cluster co anh o nhieu split khac nhau)",
        f"- Cluster bi leakage: {leakage_count}",
        f"- -> Phai re-split theo cluster de tranh data leakage",
    ]

    out_report = INTERIM_DIR / "dedup_report.md"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nTổng ảnh: {len(final_clusters)}")
    print(f"  Ảnh có augment > 2: {len(final_dup)}")
    print(f"  Ảnh đơn lẻ: {len(final_clusters) - len(final_dup)}")
    print(f"  Cross-split leakage: {leakage_count} cluster")
    print(f"  -> {out_clusters}")
    print(f"  -> {out_report}")


if __name__ == "__main__":
    main()
