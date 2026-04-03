# Trích Xuất Thông Tin CCCD Việt Nam

PoC học thuật cho bài toán trích xuất thông tin từ Căn Cước Công Dân (CCCD) Việt Nam theo hướng OCR pipeline.

## Pipeline tổng quan

```
Ảnh/Video → Card Detection → Rectification → Enhancement → Field Localization → OCR → Validate → JSON
```

## Cấu trúc dự án

```
├── data/
│   ├── cccd.v1i.coco/          # Dataset gốc từ Roboflow (COCO format)
│   │   ├── train/              # 3846 ảnh (Roboflow split gốc)
│   │   ├── valid/              # 366 ảnh
│   │   └── test/              # 187 ảnh
│   ├── interim/                # Output trung gian từ Data Quality Pipeline
│   │   ├── dedup_clusters.json
│   │   ├── dedup_report.md
│   │   ├── annotation_quality_report.md
│   │   ├── flagged_images.json
│   │   ├── dataset_statistics.json
│   │   ├── dataset_statistics_plots/
│   │   └── split_report.md
│   └── processed/
│       └── splits/             # Split sạch để train (dùng cái này)
│           ├── train.json      # 3074 ảnh (70%)
│           ├── val.json        # 699 ảnh (15%)
│           └── test.json       # 626 ảnh (15%)
│
├── scripts/
│   └── data_quality/
│       ├── dedup_images.py     # Bước 2: pHash dedup + cluster theo base name
│       ├── check_annotations.py # Bước 3: Kiểm tra chất lượng annotation
│       ├── dataset_stats.ipynb # Bước 4: Thống kê & biểu đồ phân bố
│       └── split_dataset.ipynb # Bước 5: Split 70/15/15 theo cluster
│
├── PLAN.md                     # Thiết kế chi tiết toàn bộ hệ thống
└── README.md
```

## Dataset

- **Nguồn:** [cccd by Interlock — Roboflow Universe](https://universe.roboflow.com/interlock-ihpkg/cccd-lxlem)
- **License:** CC BY 4.0
- **Số ảnh:** 4399 (từ ~769 ảnh gốc, Roboflow đã augment 5–9x)
- **Classes (7):** `card`, `id`, `name`, `birth`, `origin`, `address`, `title`
- **Limitation:** Ảnh đã qua stretch 640×640, không phải raw — ảnh hưởng ~2–5% mAP

Split Roboflow gốc bị data leakage (347/769 cluster bị tách cross-split). Dùng `data/processed/splits/` — đã re-split sạch theo cluster.

## Kết quả Data Quality

| Chỉ số | Kết quả |
|---|---|
| Tổng ảnh | 4399 |
| Clusters (ảnh gốc) | 769 |
| Ảnh có annotation lỗi | 168 (3.8%) → **PASS** < 5% |
| Split train/val/test | 3074 / 699 / 626 |
| Data leakage sau re-split | **0** |

## Lộ trình phát triển

- [x] Data Quality Pipeline (dedup, annotation check, stats, split)
- [ ] Lớp 1 — Card Detection (YOLOv8/v11, mAP@0.5 ≥ 0.85)
- [ ] Lớp 1 — Field Localization (YOLOv8/v11, mAP@0.5 ≥ 0.75)
- [ ] Module Rectification (perspective warp → 856×540)
- [ ] Module Image Enhancement (CLAHE, denoising)
- [ ] Lớp 2 — OCR (PaddleOCR cho số, VietOCR cho tiếng Việt)
- [ ] Post-processing & Validation
- [ ] Demo

Chi tiết thiết kế từng module xem tại [PLAN.md](PLAN.md).

## Citation

```bibtex
@misc{cccd-lxlem_dataset,
    title = { cccd Dataset },
    author = { Interlock },
    url = { https://universe.roboflow.com/interlock-ihpkg/cccd-lxlem },
    publisher = { Roboflow },
    year = { 2024 },
    month = { jul }
}
```
