# Vietnamese CCCD Information Extraction System

He thong trich xuat thong tin tu anh mat truoc Can cuoc cong dan Viet Nam. Du an duoc to chuc theo pipeline OCR hoan chinh: phat hien the, chuan hoa anh, phat hien tung vung thong tin, OCR, hau xu ly va danh gia ket qua.

## Trang thai hien tai

Pipeline da co the chay end-to-end cho 5 truong:

- `id_number`
- `full_name`
- `date_of_birth`
- `place_of_origin`
- `place_of_residence`

Kien truc hien tai ket hop:

- YOLO cho card detection va field detection.
- VietOCR cho nhan dang chu tieng Viet va chu so.
- PaddleOCR cho text detection/recognition bo tro, dac biet voi truong nhieu dong.
- Finetune mo hinh vietocr cho address va origin
- Parser + validator de chuan hoa ID, ngay sinh, ten va dia chi.
- Gazetteer don vi hanh chinh Viet Nam de ho tro sua loi dia danh.

## Pipeline

```text
Image
  -> Card detection
  -> Rectification / crop card
  -> Orientation correction
  -> Adaptive enhancement
  -> Field detection
  -> Field crop
  -> OCR per field
  -> Parsing / validation / normalization
  -> JSON result
```

Entry point chinh: `src/pipeline/pipeline.py`.

Vi du dung trong Python:

```python
from src.pipeline import CCCDPipeline

pipeline = CCCDPipeline(
    card_detector_path="model/card_detector/best.pt",
    field_detector_path="model/field_detector/best.pt",
    device="cpu",
)

result = pipeline("path/to/cccd.jpg")
print(result.to_dict())
```

## Cau truc project

```text
.
|-- configs/
|   |-- vietocr_finetune.yml
|   |-- vietocr_finetune_v2.yml
|   |-- vietocr_address_v1.yml
|   `-- vietocr_address_origin_reviewed.yml
|
|-- data/
|   |-- cccd.v1i.coco/                 # Dataset COCO goc
|   |-- processed/                     # Splits, OCR crops, reviewed labels, eval GT
|   |-- synthetic/                     # Du lieu tong hop cho fine-tune
|   `-- vn_administrative/             # Du lieu hanh chinh Viet Nam
|
|-- demo/
|   |-- app.py                         # Streamlit demo
|   |-- debug_extraction.py            # Debug mot anh
|   `-- test_vietocr_accuracy.py
|
|-- model/
|   |-- card_detector/best.pt          # YOLO card detector
|   `-- field_detector/best.pt         # YOLO field detector
|
|-- outputs/
|   |-- demo_debug/                    # Debug bundle tu demo
|   |-- finetune/                      # Ket qua fine-tune OCR
|   `-- full_pipeline_eval/            # Ket qua danh gia end-to-end
|
|-- scripts/
|   |-- data_quality/                  # Kiem tra annotation, dedup, split
|   |-- detection/                     # Notebook prepare/train YOLO
|   |-- finetune/                      # Chuan bi data va train VietOCR
|   |-- ocr/                           # Prepare data cho PaddleOCR/PP-OCR
|   |-- crop_fields.py                 # Crop field OCR tu annotation
|   |-- generate_pseudo_labels.py      # Tao pseudo labels OCR
|   |-- evaluate_ocr.py                # Eval OCR tren reviewed labels
|   |-- evaluate_address_origin_gt.py  # Eval rieng origin/address
|   |-- evaluate_full_pipeline_5fields.py
|   `-- ocr_performance_report.py
|
|-- src/
|   |-- preprocessing/
|   |   |-- rectify.py
|   |   |-- orientation.py
|   |   |-- enhance.py
|   |   `-- super_resolution.py
|   |-- ocr/
|   |   |-- vietocr_adapter.py
|   |   |-- paddleocr_adapter.py
|   |   |-- multiline_ocr.py
|   |   |-- text_detection.py
|   |   |-- hybrid_line_pick.py
|   |   |-- ensemble.py
|   |   |-- tta.py
|   |   |-- cropping.py
|   |   |-- utils.py
|   |   `-- types.py
|   |-- parsing/
|   |   |-- validators.py
|   |   |-- address_corrector.py
|   |   |-- vn_places.py
|   |   `-- diacritics.py
|   |-- evaluation/
|   |   |-- evaluator.py
|   |   `-- ocr_metrics.py
|   `-- pipeline/
|       `-- pipeline.py
|
|-- weights/
|   |-- vietocr_cccd_v2.pth
|   `-- vietocr_cccd_address_origin_reviewed.pth
|
|-- review_tool.py
|-- requirements.txt
|-- PLAN.md
`-- README.md
```

Ghi chu: `data/`, `model/`, `weights/`, `outputs/` va cac file checkpoint/JSONL lon dang duoc ignore trong `.gitignore`. Khi clone repo moi, can copy cac artifact nay vao dung duong dan truoc khi chay pipeline.

## Artifact can co

Pipeline mac dinh ky vong cac file sau ton tai:

```text
model/card_detector/best.pt
model/field_detector/best.pt
weights/vietocr_cccd_v2.pth
weights/vietocr_cccd_address_origin_reviewed.pth
configs/vietocr_finetune_v2.yml
configs/vietocr_address_origin_reviewed.yml
data/vn_administrative/divisions_lookup.json
```

Neu thieu weight fine-tuned, `VietOCRRecognizer` se fallback ve config/model mac dinh neu thu vien cho phep. Ket qua OCR khi fallback se khong phan anh metric hien tai.

## Cai dat

Moi truong khuyen nghi: Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install ultralytics streamlit pandas requests
```

Neu dung GPU, can cai ban `torch` va `paddlepaddle` phu hop CUDA tren may.

## Chay demo

```powershell
streamlit run demo/app.py
```

Demo cho phep upload anh hoac nhap URL anh, sau do hien thi:

- Anh goc voi card bbox.
- Bang 5 truong da trich xuat.
- Confidence tung field.
- Warning/error tu pipeline.
- Debug bundle trong `outputs/demo_debug/` neu bat tuy chon save debug.

## Debug mot anh

```powershell
python demo/debug_extraction.py path\to\image.jpg
```

Script nay in cac buoc xu ly, warning/error, OCR raw, OCR strategy, bbox crop va parsed result.

## Chuan bi du lieu OCR

Crop field tu COCO annotation:

```powershell
python scripts/crop_fields.py `
  --splits train val test `
  --output-dir data\processed\ocr\field_crops `
  --manifest-output data\processed\ocr\manifest.jsonl
```

Tao pseudo labels:

```powershell
python scripts/generate_pseudo_labels.py `
  --manifest data\processed\ocr\manifest.jsonl `
  --output data\processed\ocr\pseudo_labels.jsonl
```

Danh gia OCR tren reviewed labels:

```powershell
python scripts/evaluate_ocr.py `
  --gt data\processed\ocr\reviewed.jsonl `
  --pred data\processed\ocr\pseudo_labels.jsonl `
  --output-dir outputs\ocr_eval
```

## Danh gia full pipeline

Chay danh gia end-to-end tren 5 field:

```powershell
python scripts/evaluate_full_pipeline_5fields.py
```

Output:

```text
outputs/full_pipeline_eval/predictions.jsonl
outputs/full_pipeline_eval/metrics_summary.json
outputs/full_pipeline_eval/metrics_summary.csv
outputs/full_pipeline_eval/error_analysis.md
outputs/full_pipeline_eval/error_rows.jsonl
```

Metric hien tai trong `outputs/full_pipeline_eval/metrics_summary.json` dang duoc tinh theo che do `admin_format_only_keep_diacritics`:

- So sanh khong phan biet hoa/thuong.
- Chuan hoa Unicode NFC va khoang trang.
- `date_of_birth`: chi so sanh chu so.
- Dia chi/nguyen quan: bo dau phay va dau cau.
- Van giu dau tieng Viet, nen sai dau van bi tinh loi.

Ket qua hien tai:

| Field | Count | CER | WER | Exact match |
|---|---:|---:|---:|---:|
| `id_number` | 699 | 0.02686 | 0.10014 | 89.99% |
| `full_name` | 702 | 0.04723 | 0.13473 | 70.23% |
| `date_of_birth` | 701 | 0.02598 | 0.11698 | 88.30% |
| `place_of_origin` | 706 | 0.25859 | 0.32762 | 51.70% |
| `place_of_residence` | 699 | 0.25735 | 0.38947 | 17.17% |

Overall:

- Row-level exact: 63.47%
- CER: 0.12335
- WER: 0.21389
- Anh dung ca 5 field: 54/699, tuong duong 7.73%

## Ket qua detection

Theo `model/card_detector/eval_results.json`:

| Model | Val mAP@0.5 | Val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Card detector | 0.9950 | 0.9068 | 0.9948 | 0.9057 |
| Field detector | 0.9950 | 0.8195 | 0.9913 | 0.8070 |

Detection hien khong phai nut that lon nhat. Loi chinh nam o OCR va matching cho `place_of_origin` / `place_of_residence`.

## Uu tien cai thien tiep

1. Dua normalize `admin_format_only_keep_diacritics` vao code evaluation thay vi chi sua file summary.
2. Tach metric strict va metric relaxed trong cung mot report.
3. Cai thien `place_of_residence`, hien moi dat 17.17% exact relaxed.
4. Giam parser drift cho dia chi: parser chi nen sua khi confidence va gazetteer signal du cao.
5. Them test nho cho parser/normalizer de tranh thay doi rule lam metric nhay bat ngo.

## Citation

Dataset goc:

```bibtex
@misc{cccd-lxlem_dataset,
    title = {cccd Dataset},
    author = {Interlock},
    url = {https://universe.roboflow.com/interlock-ihpkg/cccd-lxlem},
    publisher = {Roboflow},
    year = {2024},
    month = {jul}
}
```
