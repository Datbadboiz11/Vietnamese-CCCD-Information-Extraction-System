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
│   │   └── test/               # 187 ảnh
│   ├── interim/                # Output từ Data Quality Pipeline
│   │   ├── dedup_clusters.json
│   │   ├── dedup_report.md
│   │   ├── annotation_quality_report.md
│   │   ├── flagged_images.json
│   │   ├── dataset_statistics.json
│   │   ├── dataset_statistics_plots/
│   │   └── split_report.md
│   └── processed/
│       └── splits/             # Split sạch để train
│           ├── train.json      # 3074 ảnh (70%)
│           ├── val.json        # 699 ảnh (15%)
│           └── test.json       # 626 ảnh (15%)
│
├── scripts/
│   ├── data_quality/
│   │   ├── dedup_images.py
│   │   ├── check_annotations.py
│   │   ├── dataset_stats.ipynb
│   │   └── split_dataset.ipynb
│   └── detection/
│       ├── 01_prepare_yolo_data.ipynb
│       ├── 02_train_card_detector.ipynb
│       └── 03_train_field_detector.ipynb
│
├── src/
│   └── preprocessing/
│       ├── rectify.py          # Perspective warp → 856×540
│       ├── enhance.py          # CLAHE, denoising (TODO)
│       └── orientation.py      # Rotation detection (TODO)
│
├── model/
│   ├── card_detector/
│   │   ├── best.pt             # mAP@0.5 = 0.995
│   │   └── eval_results.json
│   └── field_detector/
│       ├── best.pt             # mAP@0.5 = 0.995
│       └── eval_results.json
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

## Kết quả

### Data Quality

| Chỉ số | Kết quả |
|---|---|
| Tổng ảnh | 4399 |
| Clusters (ảnh gốc) | 769 |
| Ảnh có annotation lỗi | 168 (3.8%) → **PASS** < 5% |
| Split train/val/test | 3074 / 699 / 626 |
| Data leakage sau re-split | **0** |

### Detection

| Model | mAP@0.5 | mAP@0.5:95 | Precision | Recall |
|---|---|---|---|---|
| Card Detector (1 class) | **0.995** | 0.907 | 0.9998 | 1.000 |
| Field Detector (6 class) | **0.995** | 0.849 | 0.9995 | 0.997 |

## Lộ trình phát triển

- [x] Data Quality Pipeline (dedup, annotation check, stats, split)
- [x] Lớp 1 — Card Detection (YOLOv11s, mAP@0.5 = 0.995)
- [x] Lớp 1 — Field Localization (YOLOv11s, mAP@0.5 = 0.995)
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

## OCR Batch Commands

Quick smoke test on 20 crops:

```powershell
.\.venv\Scripts\python.exe scripts\crop_fields.py --splits val --output-dir data\interim\cropped_fields_smoke --manifest-output data\interim\cropped_fields_smoke\manifest_20.jsonl --limit-crops 20
.\.venv\Scripts\python.exe scripts\generate_pseudo_labels.py --manifest data\interim\cropped_fields_smoke\manifest_20.jsonl --output outputs\ocr_smoke\pseudo_labels_20.jsonl --limit 20
```

Full batch OCR:

```powershell
.\.venv\Scripts\python.exe scripts\crop_fields.py --splits train val test --output-dir data\interim\cropped_fields --manifest-output data\interim\cropped_fields\manifest.jsonl
.\.venv\Scripts\python.exe scripts\generate_pseudo_labels.py --manifest data\interim\cropped_fields\manifest.jsonl --output data\processed\ocr\pseudo_labels.jsonl
```

Evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ocr.py --gt data\processed\ocr\reviewed.jsonl --pred data\processed\ocr\pseudo_labels.jsonl --output-dir outputs\ocr_eval
```

Colab GPU guide:

- [COLAB_OCR_GPU.md](COLAB_OCR_GPU.md)
- [scripts/colab_batch_ocr_gpu.ipynb](scripts/colab_batch_ocr_gpu.ipynb)
