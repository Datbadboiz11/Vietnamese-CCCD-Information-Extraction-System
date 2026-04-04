# Kế Hoạch Hệ Thống Information Extraction Từ CCCD Việt Nam — v3 (Final)

## Tóm tắt

Xây hệ thống `PoC học thuật` cho bài toán trích xuất thông tin từ `CCCD Việt Nam` theo hướng `OCR pipeline`:

**Ảnh/Video gốc → Phát hiện thẻ → Hiệu chỉnh phối cảnh → Tăng cường ảnh → Xác định mặt thẻ → Định vị field → OCR → Chuẩn hóa & Validate → Đánh giá → Demo**

Dựa trên dataset [cccd trên Roboflow Universe](https://universe.roboflow.com/interlock-ihpkg/cccd-lxlem) (CC BY 4.0, by Interlock), giai đoạn v1 tập trung vào `mặt trước CCCD`. Dataset có bbox cho `card`, `id`, `name`, `birth`, `origin`, `address`, `title`, nhưng chưa có transcript OCR và chưa có field cho mặt sau.

Hệ thống chia làm 2 lớp với **tiêu chí chuyển giao rõ ràng:**

- `Lớp 1 – Detection`: card detection + field localization, dùng ngay dataset hiện có. **Chỉ chuyển sang Lớp 2 khi Detection đạt mAP@0.5 ≥ 0.85 cho card và ≥ 0.75 cho field.**
- `Lớp 2 – Recognition`: OCR + parsing, cần bổ sung transcript qua pseudo-label + review.

Hệ thống được thiết kế để hoạt động tốt trên cả **ảnh tĩnh** (chụp từ điện thoại, scan) và **video** (quay CCCD realtime). Augmentation được thiết kế theo hướng **domain-aware** — mô phỏng đúng các degradation thực tế thay vì augment ngẫu nhiên.

Đầu ra chuẩn là JSON thống nhất, mở rộng được sang mặt sau mà không đổi interface.

---

## Nguồn dữ liệu

### Dataset gốc

- **Nguồn:** [Roboflow Universe — cccd by Interlock](https://universe.roboflow.com/interlock-ihpkg/cccd-lxlem)
- **License:** CC BY 4.0 (cần cite trong báo cáo)
- **Số ảnh:** 4399
- **Classes (7):** `card`, `id`, `name`, `birth`, `origin`, `address`, `title`
- **Split Roboflow:** 3846 train (87%) / 366 valid (8%) / 187 test (4%)
- **Preprocessing Roboflow đã apply:** Auto-Orient + Resize Stretch 640×640
- **Augmentation Roboflow:** Không có

### Vấn đề với dataset Roboflow và cách xử lý

| Vấn đề | Tác động | Xử lý |
|---|---|---|
| Resize Stretch 640×640 | Méo tỷ lệ thẻ CCCD (1.59:1 → 1:1), bbox bị biến dạng nhẹ | **Quyết định: tiến hành với bản stretch 640×640 hiện có** (v1i COCO). Raw images không thể tải vì không có quyền tạo version mới. Ảnh hưởng thực tế thấp (~2–5% mAP) vì YOLO robust với aspect ratio. Ghi nhận là limitation trong báo cáo. Khi train YOLO dùng letterbox resize tại inference. |
| Test set quá nhỏ (187 ảnh, 4%) | Kết quả đánh giá không đáng tin | **Tự split lại 70/15/15** sau khi deduplicate |
| Nghi ngờ ảnh trùng lặp (cùng CCCD, nhiều biến thể) | Data leakage giữa train/test, metric bị inflate | **Chạy perceptual hash dedup** trước khi split, nhóm ảnh cùng CCCD phải nằm cùng 1 split |
| Chỉ có bbox, không có transcript | Không train OCR trực tiếp | Dùng **pseudo-label workflow** |

### Citation

```bibtex
@misc{cccd-lxlem_dataset,
    title = { cccd Dataset },
    type = { Open Source Dataset },
    author = { Interlock },
    howpublished = { \url{ https://universe.roboflow.com/interlock-ihpkg/cccd-lxlem } },
    url = { https://universe.roboflow.com/interlock-ihpkg/cccd-lxlem },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
    month = { jul }
}
```

---

## Data Quality Pipeline

Trước khi bắt đầu bất kỳ training nào, toàn bộ dataset phải đi qua quy trình kiểm tra chất lượng. Đây là bước bắt buộc — nếu bỏ qua, model sẽ học noise và metric bị inflate.

### Bước 1: Tải và chuẩn bị dữ liệu

```
1. Dataset đã tải: cccd.v1i.coco (COCO JSON, 640×640 stretch) — lưu tại data/cccd.v1i.coco/
   Gồm: train (3846 ảnh) / valid (366 ảnh) / test (187 ảnh) theo split Roboflow gốc
   Limitation: ảnh đã qua stretch 640×640, không phải raw — ghi nhận trong báo cáo
2. Kiểm tra tổng số ảnh và annotation khớp với công bố (4399 ảnh, 7 classes) ✓
3. Không chỉnh sửa gì trên thư mục data/cccd.v1i.coco/
```

### Bước 2: Kiểm tra ảnh trùng lặp (Deduplication)

```
Script: scripts/data_quality/dedup_images.py

Thuật toán:
1. Tính perceptual hash (pHash) cho mỗi ảnh
2. So sánh pairwise distance (Hamming distance ≤ 8 → coi là trùng)
3. Nhóm các ảnh trùng thành cluster
4. Log: tổng số cluster trùng, số ảnh bị trùng

Output:
- data/interim/dedup_clusters.json  (danh sách cluster)
- data/interim/dedup_report.md      (thống kê)

Xử lý:
- KHÔNG xóa ảnh trùng — giữ tất cả để train (nhiều data hơn)
- Nhưng khi split train/val/test: tất cả ảnh trong cùng 1 cluster PHẢI nằm cùng 1 split
- Điều này ngăn data leakage
```

### Bước 3: Kiểm tra annotation

```
Script: scripts/data_quality/check_annotations.py

Kiểm tra:
1. Ảnh thiếu annotation (0 bbox) → log warning
2. Ảnh thiếu class "card" → log warning (mỗi ảnh phải có ≥ 1 card)
3. Ảnh có card nhưng thiếu field (< 4 field classes) → log warning
4. Bbox quá nhỏ (diện tích < 100px²) → log error, đánh dấu để review
5. Bbox quá lớn (> 95% diện tích ảnh) → log error
6. Bbox nằm ngoài ảnh (x+w > img_width hoặc y+h > img_height) → log error
7. Bbox overlap bất thường giữa cùng class (IoU > 0.9) → log warning
8. Phân bố class: đếm số annotation mỗi class → kiểm tra imbalance

Output:
- data/interim/annotation_quality_report.md
- data/interim/flagged_images.json (ảnh cần review thủ công)

Tiêu chí pass:
- < 5% ảnh có annotation error
- Không có class nào có < 100 annotations
- Tất cả error ảnh đã được review và fix/loại bỏ
```

### Bước 4: Thống kê dataset

```
Script: scripts/data_quality/dataset_stats.py

Thống kê:
1. Phân bố kích thước ảnh (width, height, aspect ratio)
2. Phân bố số annotation per image
3. Phân bố kích thước bbox per class (width, height, area)
4. Phân bố vị trí bbox per class (center x, center y)
5. Co-occurrence matrix giữa các class
6. Histogram brightness / contrast

Output:
- data/interim/dataset_statistics.json
- data/interim/dataset_statistics_plots/  (biểu đồ phân bố)
- Dùng cho phần thống kê dữ liệu trong báo cáo
```

### Bước 5: Split dataset

```
Script: scripts/data_quality/split_dataset.py

Quy tắc:
1. Tỷ lệ: 70% train / 15% val / 15% test
2. Split theo cluster (từ bước dedup): tất cả ảnh cùng cluster → cùng split
3. Stratified theo difficulty: đảm bảo mỗi split có tỷ lệ tương đương ảnh khó/dễ
   - Difficulty score = hàm của: số annotation, kích thước bbox nhỏ nhất, brightness
4. Random seed cố định (seed=42) để reproducible
5. Verify sau khi split: không có pHash trùng cross-split

Output:
- data/processed/splits/train.json
- data/processed/splits/val.json
- data/processed/splits/test.json
- data/interim/split_report.md (thống kê mỗi split)
```

---

## Thiết kế hệ thống

### 1. Input và Output

**Input:**

- Ảnh CCCD chụp từ điện thoại hoặc scan.
- Frame trích xuất từ video quay CCCD.
- V1 giả định mỗi ảnh/frame chứa một thẻ.
- V1 ưu tiên ảnh mặt trước.
- Hỗ trợ định dạng: JPEG, PNG, MP4 (video). Khuyến nghị resolution tối thiểu 640×480.

**Output chuẩn:**

```json
{
  "pipeline_version": "1.0",
  "input_type": "image",
  "card_type": "cccd_chip",
  "side": "front",
  "image_quality_score": 0.85,
  "fields": {
    "id_number": "...",
    "full_name": "...",
    "date_of_birth": "...",
    "place_of_origin": "...",
    "place_of_residence": "..."
  },
  "confidence": {
    "id_number": 0.0,
    "full_name": 0.0,
    "date_of_birth": 0.0,
    "place_of_origin": 0.0,
    "place_of_residence": 0.0
  },
  "artifacts": {
    "warped_card_path": "...",
    "enhanced_card_path": "...",
    "field_boxes_path": "...",
    "ocr_raw_path": "..."
  },
  "processing_time_ms": 0,
  "needs_review": true,
  "review_reasons": [],
  "tta_applied": false,
  "video_meta": {
    "source_frame_idx": null,
    "total_candidate_frames": null,
    "selection_method": null
  }
}
```

---

### 2. Các module chính

#### Module 1: Card Detection

- **Input:** ảnh gốc hoặc video frame
- **Output:** bbox hoặc polygon của CCCD
- **Dữ liệu:** class `card` từ dataset
- **Mô hình:** YOLOv8 hoặc YOLOv11
- **Resize strategy:** Letterbox resize (giữ tỷ lệ, thêm padding) thay vì stretch. Kích thước input YOLO: 640×640 letterbox.
- **Lưu ý kỹ thuật:**
  - Ưu tiên predict **polygon/oriented bbox** thay vì axis-aligned bbox.
  - Nếu model chỉ ra bbox, cần thêm bước **corner refinement** để tìm 4 góc thẻ (xem Module 2).
  - Ngưỡng confidence card detection: `≥ 0.5`. Nếu thấp hơn → flag `needs_review`.

#### Module 2: Rectification

- **Input:** ảnh gốc + card bbox/polygon
- **Output:** ảnh thẻ đã warp về kích thước chuẩn (856×540 pixel, tỷ lệ CCCD 85.6mm × 54mm)
- **Chiến lược tìm 4 góc (theo thứ tự ưu tiên):**

| Phương pháp | Khi nào dùng | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Keypoint detection** (primary, nếu đủ thời gian) | Train model predict 4 corners | Robust nhất, hoạt động tốt trên ảnh khó | Cần thêm annotation 4 góc hoặc tự tạo từ bbox |
| **Contour + approxPolyDP** (secondary) | Khi card detection ra bbox | Không cần train thêm | Fail khi nền cùng màu thẻ, thẻ bị che |
| **Hough Line Detection** (fallback) | Khi contour fail | Tìm cạnh thẳng của thẻ | Fail khi nhiều đường thẳng trong ảnh |
| **Bbox crop + padding 5%** (last resort) | Khi tất cả fail | Luôn hoạt động | Không hiệu chỉnh phối cảnh |

- **Kỹ thuật warp:**
  - `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()` với 4 corner points.
  - Sắp xếp 4 điểm: top-left → top-right → bottom-right → bottom-left.
- **Fallback:** ghi `rectification_method` vào output để biết dùng phương pháp nào, ghi `rectification_failed` vào `review_reasons` nếu phải dùng last resort.

#### Module 3: Orientation Detection *(mới)*

- **Input:** ảnh thẻ đã crop/warp
- **Output:** góc xoay (0° / 90° / 180° / 270°) + ảnh đã xoay đúng hướng
- **Tại sao cần:** nếu ảnh bị xoay 90° hoặc 180°, card detection vẫn detect được nhưng rectification sẽ warp sai hướng, field localization sẽ fail hoàn toàn.
- **Phương pháp V1:** rule-based — chạy field detector trên ảnh + 3 phiên bản xoay (90°, 180°, 270°), chọn hướng có tổng confidence field cao nhất.
- **Phương pháp V2 (nếu đủ thời gian):** train classifier nhẹ (MobileNetV3-Small) phân loại 4 hướng.

#### Module 4: Image Enhancement

- **Input:** ảnh thẻ đã rectify + orientation corrected
- **Output:** ảnh thẻ đã tăng cường chất lượng + `image_quality_score`
- **Kỹ thuật:**
  - Cân bằng sáng cục bộ: CLAHE (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`)
  - Khử noise: bilateral filter hoặc `cv2.fastNlMeansDenoising()`
  - Phát hiện lóa: detect vùng overexposed (pixel > 250) → cảnh báo, giảm contrast cục bộ
  - Tính `image_quality_score`:
    - Sharpness: `cv2.Laplacian(gray, cv2.CV_64F).var()` — normalize to [0,1]
    - Brightness: `1.0 - abs(mean_pixel - 140) / 140`
    - Contrast: std of pixel values, normalize
    - Score = 0.5 × sharpness + 0.3 × brightness + 0.2 × contrast

#### Module 5: Side Classification

- **V1:** rule-based — kiểm tra output field detector:
  - Phát hiện ≥ 2 field mặt trước (`id`, `name`, `birth`) → `side = "front"`
  - Nếu không → `side = "unknown"`, flag `needs_review`
- **Sau này:** classifier nhẹ (MobileNetV3/ResNet18) nếu có data mặt sau.

#### Module 6: Field Localization

- **Input:** ảnh thẻ đã rectify + enhance
- **Output:** bbox từng field

| Hướng | Khi nào dùng | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Field Detector** (primary) | Mặc định | Robust với rectification không hoàn hảo | Cần train model |
| **Template Crop** (fallback) | Khi detector fail hoặc confidence < 0.3 | Đơn giản, nhanh | Phụ thuộc nặng vào chất lượng rectification |

- **Quyết định:** dùng field detector làm primary vì dataset đã có annotation sẵn.
- Mô hình: YOLOv8/YOLOv11, class `id`, `name`, `birth`, `origin`, `address`, `title`.
- Resize: letterbox 640×640, giống card detector.

#### Module 7: OCR

*(Xem section "Chiến lược OCR" bên dưới để biết chi tiết đầy đủ)*

- **Input:** crop của từng field
- **Output:** `{text, confidence}`

#### Module 8: Post-processing / Parsing

- **Mapping class → schema chuẩn:**
  - `id` → `id_number`
  - `name` → `full_name`
  - `birth` → `date_of_birth`
  - `origin` → `place_of_origin`
  - `address` → `place_of_residence`

- **Rule validate (thiết kế mềm — cảnh báo thay vì reject cứng):**

| Field | Rule | Hard reject | Soft warning | Ví dụ auto-correct |
|---|---|---|---|---|
| `id_number` | Regex `^\d{12}$` | Không phải 12 ký tự | Prefix không khớp mã tỉnh | O→0, B→8, l→1, S→5 |
| `date_of_birth` | Format DD/MM/YYYY, ngày hợp lệ | Ngày không tồn tại (32/13/...) | Năm ngoài 1900–2025 (cảnh báo, không reject) | — |
| `full_name` | Chứa ký tự Unicode Vietnamese + space | Chứa số hoặc ký tự đặc biệt | Không viết hoa, quá ngắn (< 2 từ) | NGUYEN VAN A → title case |
| `place_of_origin` | Text, fuzzy match tỉnh/huyện/xã VN | — | Không match được địa danh nào (cảnh báo) | fuzzy match gần nhất |
| `place_of_residence` | Tương tự origin | — | Tương tự | fuzzy match gần nhất |

  **Lưu ý:** quy tắc năm sinh 1900–2025 là soft warning, không phải hard reject — tránh loại nhầm edge case (người rất già, trẻ sơ sinh). Tên người dân tộc thiểu số có thể chứa ký tự đặc biệt → cảnh báo thay vì reject.

- **Confidence-based routing:**
  - ≥ 0.8: accept
  - 0.5–0.8: accept + flag `needs_review` + ghi reason
  - < 0.5: thử OCR retry với ảnh enhance khác (sharpen, rotate ±2°). Nếu vẫn < 0.5 → `field_value = null`, flag review

#### Module 9: Evaluation

*(Xem section "Test Plan" để biết chi tiết metrics)*

---

## Chiến lược OCR — Chi tiết

### Phân tách theo loại field

Các field trên CCCD có bản chất rất khác nhau — dùng chung 1 model 1 config cho tất cả sẽ không tối ưu.

| Nhóm field | Fields | Đặc điểm text | Model khuyến nghị | Lý do |
|---|---|---|---|---|
| **Số thuần** | `id_number` | 12 chữ số, font cố định | PaddleOCR (recognition only) + regex validate | Đơn giản, chỉ cần nhận dạng số |
| **Ngày tháng** | `date_of_birth` | DD/MM/YYYY, chủ yếu số + `/` | PaddleOCR | Format cố định, dễ validate |
| **Tiếng Việt** | `full_name`, `place_of_origin`, `place_of_residence` | Tiếng Việt có dấu, độ dài biến thiên, ký tự đặc biệt | VietOCR (Transformer-based) | Cần model hiểu dấu tiếng Việt, context ngữ nghĩa |

### Thứ tự ưu tiên fine-tune

Nếu thời gian có hạn, ưu tiên fine-tune theo thứ tự:

1. **`id_number`** — field quan trọng nhất, dễ đánh giá exact match, dễ fine-tune vì chỉ có 10 ký tự số.
2. **`full_name`** — field quan trọng thứ 2, tiếng Việt có dấu cần fine-tune nhất.
3. **`date_of_birth`** — format chuẩn, pretrained OCR thường đã đủ tốt.
4. **`place_of_origin`, `place_of_residence`** — text dài, phức tạp, fine-tune nếu còn thời gian.

### Chiến lược tạo OCR dataset (pseudo-label workflow)

```
Bước 1: Crop tất cả field từ bbox annotation có sẵn
  - Script: scripts/crop_fields.py
  - Input: ảnh gốc + COCO annotations
  - Output: data/interim/cropped_fields/{class}/{image_id}_{class}_{idx}.jpg
  - Ước tính: ~4000 ảnh × ~6 field = ~24000 crops

Bước 2: Chạy OCR pretrained tạo pseudo-label
  - Script: scripts/generate_pseudo_labels.py
  - Chạy VietOCR + PaddleOCR trên tất cả crops
  - Lấy kết quả confidence cao nhất giữa 2 model
  - Output: data/processed/ocr/pseudo_labels.jsonl
    Format: {"crop_path": "...", "text_vietocr": "...", "conf_vietocr": 0.0,
             "text_paddleocr": "...", "conf_paddleocr": 0.0, "best_text": "...", "best_conf": 0.0}

Bước 3: Phân loại pseudo-label theo confidence
  - High confidence (≥ 0.9): accept trực tiếp, chỉ spot-check 10%
  - Medium confidence (0.5–0.9): review thủ công
  - Low confidence (< 0.5): gán thủ công từ đầu
  - Ước tính giảm ~60-70% công sức so với gán hoàn toàn thủ công

Bước 4: Review tool
  - Script đơn giản hiển thị crop + pseudo-label, người review: Accept / Sửa / Skip
  - Output: data/processed/ocr/reviewed.jsonl

Bước 5 (optional): Fine-tune
  - Dùng reviewed dataset fine-tune VietOCR
  - Ưu tiên fine-tune trên field có CER cao nhất trước
```

### Xử lý khi OCR fail

| Tình huống | Hành động |
|---|---|
| OCR confidence < 0.5 lần 1 | Retry với ảnh sharpen (Unsharp Mask) |
| Retry vẫn < 0.5 | Retry với CLAHE enhance + rotate ±2° |
| Vẫn fail | `field_value = null`, flag `needs_review`, ghi `"ocr_failed"` vào `review_reasons` |
| 2 OCR model cho kết quả khác nhau | Chọn kết quả có confidence cao hơn. Nếu confidence gần nhau (chênh < 0.1): chọn kết quả pass validation rule |

### OCR Ensemble Strategy

Chạy song song VietOCR + PaddleOCR trên mỗi field crop, ensemble kết quả:

```
Ensemble logic:
1. Nếu 2 model cho cùng kết quả → accept (confidence = max của 2)
2. Nếu khác nhau:
   a. Chọn kết quả pass validation rule (ví dụ: id_number đúng 12 số)
   b. Nếu cả 2 đều pass hoặc cả 2 đều fail → chọn confidence cao hơn
   c. Nếu confidence chênh < 0.1 → flag needs_review
```

---

## Data Augmentation — Chiến lược Domain-Aware cho Real-World & Video

### Phân tích gap giữa dataset training và thực tế

Dataset Roboflow (4399 ảnh) thường được chụp trong điều kiện tương đối kiểm soát. Ảnh thực tế và video frame có nhiều degradation khác biệt:

| Degradation thực tế | Tần suất | Ảnh hưởng | Augmentation cover |
|---|---|---|---|
| Motion blur (tay rung, thẻ di chuyển) | Rất cao (video), cao (ảnh) | Card detection giảm, OCR sai | Nhóm 3: MotionBlur, Directional MotionBlur |
| Lóa phản chiếu từ chip NFC / laminate | Rất cao | Field bị mất text, OCR fail | Nhóm 4: Specular Highlight, Local Overexposure |
| Ánh sáng không đều (1 bên sáng, 1 bên tối) | Cao | OCR thiên lệch theo vùng | Nhóm 2: Uneven Lighting, RandomShadow |
| Góc nghiêng lớn (> 30°) | Cao | Rectification khó | Nhóm 1: Rotation, Perspective |
| JPEG compression nặng (chụp từ app, gửi qua chat) | Cao | Text bị artifact, nhòe ký tự | Nhóm 3: ImageCompression |
| Nền lộn xộn (bàn gỗ, giấy tờ, ví, bàn phím) | Trung bình–Cao | Card detection nhầm nền | Nhóm 5: Background Replacement |
| Resolution thấp (video 480p, zoom xa) | Trung bình | Text quá nhỏ, OCR fail | Nhóm 3: Downscale→Upscale |
| Color cast (đèn vàng, huỳnh quang, LED) | Trung bình | Ảnh hưởng contrast text | Nhóm 2: HueSaturationValue, ColorJitter |
| Ngón tay / vật che thẻ | Trung bình | Field bị mất, detection thiếu | Nhóm 4: Finger Overlay, CoarseDropout |
| Chụp CCCD trên màn hình (screenshot, chụp màn hình khác) | Trung bình | Moiré pattern, pixel grid, color shift | Nhóm 3: Moiré Simulation |
| Thẻ cũ bị trầy xước, bong laminate, mờ text | Trung bình | OCR sai ở vùng hư | Nhóm 4: Scratch Simulation, Local Fade |
| Nhiều thẻ trên cùng bề mặt | Thấp–Trung bình | Card detection nhầm thẻ khác | Nhóm 5: Multi-card Distraction |
| Ảnh grayscale / B&W (app scan tài liệu) | Thấp–Trung bình | Model không quen ảnh không màu | Nhóm 2: ToGray, Channel Shuffle |
| Overexposure cục bộ (flash gần, nắng chiếu) | Trung bình | 20-40% ảnh bị trắng xóa | Nhóm 4: Local Overexposure Patch |
| Interlacing / rolling shutter (video) | Thấp | Sọc ngang, biến dạng dọc | Video aug: Interlace, Rolling Shutter |

### Augmentation Pipeline — 6 nhóm

Framework: **Albumentations** làm nền tảng (hỗ trợ transform bbox cùng ảnh). Các augmentation custom implement dưới dạng `albumentations.ImageOnlyTransform` hoặc `DualTransform` (nếu cần transform bbox theo).

#### Nhóm 1 — Geometric (góc chụp & framing)

| Augmentation | Tham số | Mục đích | p |
|---|---|---|---|
| Rotation | ±20° | Ảnh chụp nghiêng | 0.5 |
| Perspective | scale 0.05–0.10 | Méo phối cảnh khi cầm tay | 0.4 |
| Affine (shear) | ±10° | Biến dạng do góc chụp lệch | 0.3 |
| RandomCrop | 80–95% ảnh | Frame video không center thẻ | 0.3 |
| ShiftScaleRotate | shift ±5%, scale ±10% | Thẻ không nằm giữa ảnh | 0.4 |

#### Nhóm 2 — Photometric (ánh sáng & màu sắc)

| Augmentation | Tham số | Mục đích | p |
|---|---|---|---|
| RandomBrightnessContrast | brightness ±40%, contrast ±40% | Ảnh quá sáng / quá tối | 0.6 |
| RandomGamma | gamma 0.5–2.0 | Under/overexposed | 0.3 |
| HueSaturationValue | hue ±10, sat ±30, val ±30 | Color cast đèn vàng/xanh/LED | 0.4 |
| CLAHE | clip_limit 1–4 | Cân bằng histogram local | 0.3 |
| RandomShadow | 1–2 shadows, intensity 0.3–0.7 | Bóng đổ lên thẻ (tay, vật) | 0.3 |
| RandomSunFlare | angle (0,1), intensity (0.1,0.3) | Lóa nắng / đèn flash | 0.2 |
| ColorJitter | brightness 0.2, contrast 0.2, sat 0.2 | Thay đổi màu tổng thể | 0.3 |
| Uneven Lighting *(custom)* | gradient linear/radial, multiply 0.5–1.5 | Ánh sáng không đều (1 bên sáng 1 bên tối) | 0.3 |
| ToGray | — | Ảnh grayscale từ app scan tài liệu | 0.1 |
| ChannelShuffle | — | Hoán đổi kênh màu, tăng color robustness | 0.05 |

#### Nhóm 3 — Degradation (chất lượng ảnh kém)

| Augmentation | Tham số | Mục đích | p |
|---|---|---|---|
| GaussianBlur | kernel 3–7 | Mất nét (focus sai) | 0.3 |
| MotionBlur | kernel 5–15 | Tay rung, thẻ di chuyển | 0.4 |
| GaussNoise | var 10–50 | Noise từ camera | 0.3 |
| ISONoise | color_shift (0.01,0.05), intensity (0.1,0.5) | ISO cao (chụp thiếu sáng) | 0.3 |
| ImageCompression (JPEG) | quality 30–70 | JPEG artifact từ app chụp/gửi qua chat | 0.4 |
| Downscale→Upscale | scale 0.3–0.7 | Resolution thấp (video 480p, zoom xa) | 0.3 |
| Posterize | num_bits 4–6 | Color banding (video bitrate thấp) | 0.1 |
| Moiré Simulation *(custom)* | sine wave overlay, freq 20-60px, alpha 0.1-0.3 | Chụp CCCD trên màn hình | 0.15 |

#### Nhóm 4 — Occlusion & Surface Damage (bị che, phản xạ, hư hỏng)

| Augmentation | Tham số | Mục đích | p |
|---|---|---|---|
| CoarseDropout | max 3 holes, 10% size | Vật che ngẫu nhiên | 0.3 |
| Specular Highlight *(custom)* | ellipse trắng, alpha 0.4–0.8, 5-15% diện tích | Lóa chip NFC / lớp laminate | 0.3 |
| Local Overexposure Patch *(custom)* | gradient trắng lớn, 20-40% diện tích, alpha 0.5-0.9 | Flash chụp gần, nắng chiếu qua cửa | 0.2 |
| Finger Overlay *(custom)* | rectangle màu da, 5-12% cạnh, 1-2 cạnh | Ngón tay cầm thẻ che mép | 0.2 |
| Scratch Lines *(custom)* | 2-5 đường mỏng random, alpha 0.3-0.6 | Thẻ cũ bị trầy xước | 0.15 |
| Local Fade *(custom)* | vùng oval 10-25% diện tích, giảm contrast 40-70% | Thẻ cũ bong laminate, mờ text cục bộ | 0.15 |
| GridDropout | ratio 0.1–0.2 | Artifact scan / photocopy | 0.1 |

#### Nhóm 5 — Context & Background (nền & bối cảnh) *(nhóm mới)*

| Augmentation | Tham số | Mục đích | p |
|---|---|---|---|
| Background Replacement *(custom)* | paste card lên texture random (gỗ, vải, giấy, bàn phím...) | Nền đa dạng thay vì nền sạch | 0.3 |
| Multi-card Distraction *(custom)* | paste 1-2 thẻ random (bằng lái, thẻ ngân hàng, thẻ nhân viên) vào ảnh | Model phân biệt CCCD với thẻ khác | 0.15 |
| Edge Clutter *(custom)* | thêm đối tượng random (bút, kẹp giấy, cốc) ở viền ảnh | Nền bàn làm việc thực tế | 0.1 |

#### Nhóm 6 — Video-Specific

| Augmentation | Tham số | Mục đích | p |
|---|---|---|---|
| Directional MotionBlur *(custom)* | kernel 7–21, angle cố định per image | Motion blur có hướng (khác GaussianBlur đều) | 0.4 |
| Low Bitrate Simulation | JPEG quality 20–40 + Posterize 3–5 bit | Video bitrate thấp | 0.3 |
| Rolling Shutter *(custom)* | shear theo chiều dọc, offset dần | CMOS rolling shutter artifact | 0.1 |
| Frame Interlacing *(custom)* | xen kẽ dòng chẵn/lẻ, offset 1-3px | Video interlaced cũ | 0.1 |

### Custom Augmentation — Chi tiết implement

Tất cả custom augmentation implement dưới dạng class kế thừa `albumentations.ImageOnlyTransform` (hoặc `DualTransform` nếu cần transform bbox), lưu trong `src/data/custom_augmentations.py`.

**a) Specular Highlight (lóa chip NFC)**
```
Thuật toán:
1. Random vị trí ellipse (ưu tiên vùng chip: góc phải dưới mặt trước CCCD)
2. Tạo gradient ellipse trắng, kích thước 5-15% diện tích thẻ
3. Apply gaussian blur (kernel 15-25) cho vùng highlight để mềm viền
4. Blend với alpha 0.4-0.8 lên ảnh gốc
Tham số: max_highlights=2, size_range=(0.05, 0.15), alpha_range=(0.4, 0.8)
```

**b) Local Overexposure Patch (flash / nắng chiếu)**
```
Thuật toán:
1. Chọn 1-2 vị trí random trên ảnh
2. Tạo gradient ellipse/rectangle lớn (20-40% diện tích ảnh)
3. Gradient từ trắng (center) đến transparent (edge)
4. Blend alpha 0.5-0.9
Khác Specular Highlight ở: diện tích lớn hơn nhiều, mô phỏng flash/nắng thay vì phản chiếu chip
Tham số: size_range=(0.2, 0.4), alpha_range=(0.5, 0.9)
```

**c) Finger Overlay (ngón tay che thẻ)**
```
Thuật toán:
1. Chọn random 1-2 cạnh (trái/phải/trên/dưới)
2. Tạo hình ngón tay: rounded rectangle, chiều rộng 5-12% cạnh thẻ
3. Màu skin-tone: random trong range HSV(0-25, 50-170, 100-230)
4. Edge feathering: gaussian blur 3-7px ở viền
5. Thêm variation: slight rotation, tapering (ngón tay hẹp dần)
Tham số: num_fingers=1-2, width_range=(0.05, 0.12), sides=['left','right','bottom']
```

**d) Uneven Lighting (ánh sáng không đều)**
```
Thuật toán:
1. Chọn random hướng gradient (0°-360°)
2. Tạo gradient linear: 1 phía multiply 1.2-1.5 (sáng), phía kia multiply 0.5-0.8 (tối)
3. Hoặc radial gradient: center sáng, viền tối (hoặc ngược lại)
4. Apply multiplicative blending
Tham số: light_factor=(1.2, 1.5), dark_factor=(0.5, 0.8), gradient_type=['linear','radial']
```

**e) Scratch Lines (vết trầy xước)**
```
Thuật toán:
1. Random 2-5 đường thẳng hoặc cong nhẹ (Bezier curve)
2. Độ dày 1-3px, màu trắng hoặc xám nhạt (200-255)
3. Alpha 0.3-0.6
4. Hướng random nhưng ưu tiên ngang (mô phỏng rút thẻ ra/vào ví)
Tham số: num_scratches=(2, 5), thickness=(1, 3), alpha_range=(0.3, 0.6)
```

**f) Local Fade (bong laminate, mờ text)**
```
Thuật toán:
1. Chọn 1-2 vùng oval random, kích thước 10-25% diện tích thẻ
2. Trong vùng đó: giảm contrast 40-70% (đẩy pixel về mean)
3. Thêm slight gaussian blur (kernel 3-5) trong vùng
4. Feather viền vùng fade
Tham số: num_patches=(1, 2), size_range=(0.1, 0.25), contrast_reduction=(0.4, 0.7)
```

**g) Background Replacement (thay nền)**
```
Thuật toán:
1. Cần card mask (có thể tạo từ card bbox: tạo mask hình chữ nhật theo bbox)
2. Crop card region từ ảnh gốc theo mask
3. Chọn random background từ tập texture (thu thập 50-100 ảnh nền: bàn gỗ, vải, giấy, bàn phím, granite...)
4. Resize background cho khớp kích thước ảnh
5. Paste card lên background mới
6. Thêm slight shadow/edge transition cho realistic
Lưu ý: bbox annotation giữ nguyên vì card vẫn ở cùng vị trí
Tham số: texture_dir='data/textures/', shadow_alpha=0.2
Thu thập texture: download 50-100 ảnh surface texture từ internet hoặc tự chụp
```

**h) Multi-card Distraction (thêm thẻ khác)**
```
Thuật toán:
1. Thu thập 20-30 ảnh thẻ khác (bằng lái, thẻ ngân hàng, thẻ nhân viên — có thể dùng ảnh fake/template)
2. Random resize thẻ distraction (80-120% kích thước card chính)
3. Paste vào vùng NGOÀI card bbox (không overlap card chính)
4. Random rotation nhẹ (±15°)
Lưu ý: KHÔNG thêm annotation cho thẻ distraction — model chỉ cần detect card CCCD
Tham số: distraction_dir='data/distraction_cards/', max_cards=2
```

**i) Moiré Simulation (chụp qua màn hình)**
```
Thuật toán:
1. Tạo sine wave pattern 2D: sin(x*freq) * sin(y*freq)
2. Frequency: 20-60 pixels per cycle
3. Rotation angle random
4. Overlay lên ảnh với alpha 0.1-0.3
5. Thêm slight color shift (blue tint) để mô phỏng màn hình
Tham số: freq_range=(20, 60), alpha_range=(0.1, 0.3), color_shift=True
```

**j) Directional Motion Blur (motion blur có hướng)**
```
Thuật toán:
1. Tạo kernel motion blur với angle cố định (khác MotionBlur thông thường có angle random per pixel)
2. Kernel size 7-21
3. Angle random 1 lần per image (mô phỏng camera/thẻ di chuyển theo 1 hướng)
Tham số: kernel_range=(7, 21)
```

### Augmentation cho OCR Crop (tầng field)

Ngoài augmentation ở tầng ảnh gốc (cho detection), cần augmentation riêng cho OCR crop vì đặc thù khác:

| Augmentation | Tham số | Mục đích | p |
|---|---|---|---|
| ElasticTransform | alpha 30–50, sigma 4–6 | Biến dạng nhẹ ký tự (ink spread) | 0.3 |
| RandomScale | 0.8–1.2 | Kích thước chữ khác nhau | 0.3 |
| Erosion | kernel 1–3 | Chữ mỏng hơn (mực nhạt, in mờ) | 0.2 |
| Dilation | kernel 1–3 | Chữ đậm hơn (mực dày, in đậm) | 0.2 |
| Partial Erase *(custom)* | xóa 5–15% vùng text, random rectangular | Ký tự bị mất một phần (trầy xước, bong laminate) | 0.15 |
| Line Through Text *(custom)* | 1-2 đường ngang, alpha 0.2-0.4 | Vạch/nếp gấp ngang qua text | 0.1 |
| Background Noise Injection *(custom)* | thêm texture nền thẻ (họa tiết, viền, ký hiệu) vào crop | Crop thực tế không có nền sạch | 0.3 |
| RandomBrightnessContrast | ±30% | Crop từ vùng sáng/tối khác nhau | 0.4 |
| GaussNoise | var 5–30 | Noise nhỏ ảnh hưởng ký tự | 0.2 |
| ImageCompression | quality 40–80 | JPEG artifact trên crop | 0.3 |

**Chi tiết custom OCR augmentation:**

**Background Noise Injection:**
```
Thuật toán:
1. Thu thập 20-30 mẫu nền thẻ CCCD (vùng không có text: họa tiết hoa văn, viền trang trí, vùng nền gradient)
2. Random chọn 1 mẫu, resize về kích thước crop
3. Blend nhẹ (alpha 0.1-0.3) dưới text
4. Hoặc: thêm random pattern (dot, line, texture) vào vùng trống của crop
Tham số: bg_samples_dir='data/bg_textures/', alpha_range=(0.1, 0.3)
```

**Line Through Text:**
```
Thuật toán:
1. 1-2 đường ngang (y random trong 30-70% chiều cao crop)
2. Màu random: trắng (nếp gấp) hoặc đen (vết bút)
3. Thickness 1-2px, alpha 0.2-0.4
4. Slight wave (sine offset ±2px) thay vì đường thẳng hoàn hảo
```

### Quy tắc Compose Augmentation

```
Training Detection Pipeline:
  OneOf([
    LightAugment(p=0.35),      # 1-2 aug từ nhóm 1+2
    MediumAugment(p=0.35),     # 2-3 aug từ nhóm 1+2+3
    HeavyAugment(p=0.20),      # 3-4 aug từ nhóm 1+2+3+4
    ExtremeAugment(p=0.10),    # 5-6 aug từ tất cả nhóm (worst case)
  ])
  + OneOf([                     # Background aug (independent)
    NoBackgroundChange(p=0.7),
    BackgroundReplacement(p=0.2),
    MultiCardDistraction(p=0.1),
  ])
  + Normalize(mean, std)
  + LetterboxResize(640×640)

Training OCR Pipeline:
  SomeOf(n=2-3, [              # Chọn 2-3 từ danh sách
    ElasticTransform,
    Erosion/Dilation,
    PartialErase,
    LineThrough,
    BackgroundNoise,
    RandomBrightnessContrast,
    GaussNoise,
    ImageCompression,
  ])
  + Normalize
  + Resize(target_height=32 or 64)
```

**Chi tiết từng level:**

| Level | Số aug đồng thời | Nguồn nhóm | Tình huống mô phỏng |
|---|---|---|---|
| Light | 1–2 | Nhóm 1 + 2 | Ảnh chụp bình thường, hơi nghiêng hoặc hơi tối |
| Medium | 2–3 | Nhóm 1 + 2 + 3 | Ảnh chất lượng trung bình: mờ nhẹ + tối + compression |
| Heavy | 3–4 | Nhóm 1 + 2 + 3 + 4 | Ảnh khó: nghiêng + tối + mờ + ngón tay che |
| Extreme | 5–6 | Tất cả nhóm | Worst case: nghiêng + tối + mờ + lóa + che + nền lộn xộn |

**Quy tắc quan trọng:**
- **Chỉ augment trên train set.** Val/test giữ nguyên tuyệt đối.
- **Background augmentation** (nhóm 5) apply **độc lập** với các nhóm khác — không nằm trong OneOf chính, vì thay đổi nền không conflict với các degradation khác.
- **Video augmentation** (nhóm 6) chỉ apply khi training cho video-specific model hoặc khi muốn tăng robustness trên video frame. Mặc định: include vào pool chung với p thấp.
- **Monitor:** nếu val mAP giảm > 5% so với no-augmentation → giảm Extreme từ 10% xuống 5%, giảm p của các augmentation nặng nhất.
- **Augmentation config** lưu riêng trong `configs/augmentation.yaml` để dễ ablation study.

### Tài nguyên cần thu thập cho Custom Augmentation

| Tài nguyên | Số lượng | Cách thu thập | Dùng cho |
|---|---|---|---|
| Background textures (bàn gỗ, vải, giấy, granite, bàn phím...) | 50–100 ảnh | Download từ Unsplash/Pexels (free license) hoặc tự chụp | Background Replacement |
| Distraction cards (bằng lái, thẻ ngân hàng, thẻ nhân viên...) | 20–30 ảnh | Dùng template/fake cards, KHÔNG dùng thẻ thật có thông tin | Multi-card Distraction |
| CCCD background textures (họa tiết nền thẻ) | 10–20 mẫu crop | Crop từ dataset hiện có (vùng không có text) | OCR Background Noise |

**Lưu trong:** `data/augmentation_assets/{textures, distraction_cards, bg_textures}/`
**Thêm vào `.gitignore`** nếu file lớn, commit danh sách URL download thay vì file gốc.

---

## Test-Time Augmentation (TTA)

TTA tăng robustness khi inference trên ảnh khó mà không cần train lại.

### TTA cho Detection

```
Input ảnh gốc
  ├─ Original → detect → boxes_0
  ├─ Flip horizontal → detect → boxes_1 (flip lại tọa độ)
  ├─ Scale 0.8x → detect → boxes_2 (scale lại tọa độ)
  └─ Scale 1.2x → detect → boxes_3 (scale lại tọa độ)

Ensemble: Weighted Boxes Fusion (WBF) hoặc NMS
```

### TTA cho OCR

```
Input field crop
  ├─ Original → OCR → text_0, conf_0
  ├─ Sharpen → OCR → text_1, conf_1
  ├─ CLAHE enhance → OCR → text_2, conf_2
  └─ Rotate ±2° → OCR → text_3, conf_3

Ensemble: max confidence hoặc majority vote (≥ 2/4 giống nhau)
```

### Khi nào bật TTA

| Điều kiện | Bật TTA cho |
|---|---|
| `image_quality_score` < 0.5 | Cả detection + OCR |
| Detection confidence < 0.7 | Detection only |
| OCR confidence < 0.5 (lần 1) | OCR only |
| Ảnh đẹp, confidence cao | **Không bật** (tiết kiệm thời gian) |

### Time budget

| Mode | Target GPU | Target CPU |
|---|---|---|
| Không TTA | < 2s/ảnh | < 10s/ảnh |
| Có TTA | < 6s/ảnh | < 30s/ảnh |
| Video (10s clip) | < 30s | < 120s |

---

## Video Inference Pipeline

### Frame Selection Strategy

```
Video input (N frames)
  │
  ├─ Bước 1: Sample mỗi 5-10 frame (N → N/5~N/10)
  │     Adaptive: nếu video < 3s → sample mỗi 3 frame
  │               nếu video 3-10s → sample mỗi 5 frame
  │               nếu video > 10s → sample mỗi 10 frame
  │
  ├─ Bước 2: Card detection → lọc frames có card conf ≥ 0.5
  │
  ├─ Bước 3: Quality scoring (sharpness + brightness + motion blur)
  │
  ├─ Bước 4: Chọn top-K frames (K=3-5) quality cao nhất
  │
  ├─ Bước 5: Full pipeline trên K frames
  │
  └─ Bước 6: Ensemble
       ├─ Mỗi field: chọn OCR result confidence cao nhất
       ├─ Hoặc majority vote nếu ≥ 2/K giống nhau
       └─ Ghi video_meta
```

### Edge cases

| Tình huống | Xử lý |
|---|---|
| Video < 1 giây | Chạy trên tất cả frame |
| Tất cả frame quality kém (score < 0.3) | Output `"video_quality_insufficient"`, chọn frame tốt nhất vẫn chạy, flag `needs_review` |
| Video quay nhiều thẻ lần lượt | V1: chỉ lấy thẻ xuất hiện nhiều frame nhất. V2: detect card transitions, tách thành segments |
| Không detect được card ở frame nào | Output error: `"no_card_detected_in_video"` |

---

## Error Recovery Strategy

Decision tree khi pipeline gặp lỗi ở bất kỳ bước nào:

```
Card Detection
  ├─ Detect thành công (conf ≥ 0.5) → tiếp tục
  ├─ Detect conf thấp (0.3–0.5) → bật TTA detection, nếu vẫn thấp → tiếp tục + flag review
  └─ Không detect được → STOP, output error "no_card_detected"

Rectification
  ├─ Tìm được 4 góc → perspective warp → tiếp tục
  ├─ Tìm được 3 góc → estimate góc thứ 4 từ tỷ lệ thẻ → warp → flag review
  ├─ Tìm được < 3 góc → crop bbox + padding → tiếp tục ở degraded mode
  └─ Flag "rectification_quality" = "good" / "estimated" / "degraded"

Orientation Detection
  ├─ Confidence cao (1 hướng rõ ràng hơn hẳn) → xoay → tiếp tục
  └─ Không rõ hướng → giữ nguyên + flag review

Field Localization
  ├─ Detector tìm được ≥ 4/6 fields → tiếp tục
  ├─ Detector tìm được 1-3 fields → thử template crop cho field thiếu → tiếp tục
  ├─ Detector tìm 0 field + rectification tốt → thử template crop toàn bộ
  └─ Detector tìm 0 field + rectification degraded → STOP, output partial result

OCR
  ├─ Confidence ≥ 0.5 → accept (flag review nếu < 0.8)
  ├─ Confidence < 0.5 → retry với enhance variants
  ├─ Retry vẫn < 0.5 → field_value = null + flag review
  └─ 2 model conflict → ensemble logic (xem section OCR)
```

**Nguyên tắc:** pipeline luôn cố gắng cho ra kết quả (dù partial) thay vì dừng hoàn toàn. Flag `review_reasons` sẽ ghi rõ bước nào bị degraded để người dùng biết mức độ tin cậy.

---

## Cấu trúc thư mục hệ thống

```text
id-card-ie/
├─ configs/
│  ├─ train_detector.yaml
│  ├─ train_field_detector.yaml
│  ├─ train_ocr.yaml
│  ├─ inference.yaml
│  ├─ augmentation.yaml
│  └─ tta.yaml
├─ data/
│  ├─ raw/                              # Dữ liệu gốc Roboflow, KHÔNG chỉnh sửa
│  ├─ interim/
│  │  ├─ dedup_clusters.json            # Kết quả dedup
│  │  ├─ dedup_report.md
│  │  ├─ annotation_quality_report.md
│  │  ├─ flagged_images.json
│  │  ├─ dataset_statistics.json
│  │  ├─ dataset_statistics_plots/
│  │  ├─ split_report.md
│  │  └─ cropped_fields/               # Field crops từ bbox
│  ├─ processed/
│  │  ├─ splits/                        # train.json, val.json, test.json
│  │  ├─ detector/                      # Manifest card detector
│  │  ├─ field_detector/                # Manifest field detector
│  │  ├─ ocr/                           # OCR dataset
│  │  │  ├─ pseudo_labels.jsonl
│  │  │  └─ reviewed.jsonl
│  │  └─ e2e/                           # Manifest pipeline IE
│  ├─ annotations/
│  └─ augmentation_assets/              # Tài nguyên cho custom augmentation
│     ├─ textures/                      # 50-100 ảnh nền (bàn gỗ, vải, giấy...)
│     ├─ distraction_cards/             # 20-30 ảnh thẻ khác (fake/template)
│     └─ bg_textures/                   # 10-20 mẫu nền thẻ CCCD (crop từ dataset)
├─ scripts/
│  ├─ data_quality/
│  │  ├─ dedup_images.py
│  │  ├─ check_annotations.py
│  │  ├─ dataset_stats.py
│  │  └─ split_dataset.py
│  ├─ convert_coco_dataset.py
│  ├─ crop_fields.py
│  ├─ generate_pseudo_labels.py
│  ├─ train_detector.py
│  ├─ train_field_detector.py
│  ├─ train_ocr.py
│  ├─ run_infer.py
│  ├─ run_infer_video.py
│  └─ evaluate.py
├─ src/
│  ├─ common/                           # Logger, config loader, utils
│  ├─ data/
│  │  ├─ dataset.py
│  │  ├─ transforms.py                  # Albumentations pipelines
│  │  └─ custom_augmentations.py        # Specular, finger, uneven light
│  ├─ detection/                        # Card detector + field detector
│  ├─ preprocessing/
│  │  ├─ rectify.py                     # Corner finding + perspective warp
│  │  ├─ orientation.py                 # 0°/90°/180°/270° detection
│  │  ├─ enhance.py                     # CLAHE, denoise, quality score
│  │  └─ side_classifier.py
│  ├─ ocr/
│  │  ├─ vietocr_adapter.py
│  │  ├─ paddleocr_adapter.py
│  │  └─ ensemble.py                    # OCR ensemble logic
│  ├─ parsing/
│  │  ├─ field_mapping.py
│  │  ├─ validators.py                  # Soft/hard rules
│  │  └─ confidence_router.py
│  ├─ pipeline/
│  │  ├─ image_pipeline.py
│  │  ├─ video_pipeline.py
│  │  ├─ tta.py
│  │  └─ error_recovery.py             # Decision tree logic
│  └─ evaluation/
│      ├─ detection_metrics.py
│      ├─ ocr_metrics.py
│      ├─ e2e_metrics.py
│      ├─ error_analysis.py
│      └─ augmentation_ablation.py
├─ demo/
├─ docs/
│  ├─ annotation_guide.md
│  ├─ augmentation_guide.md
│  ├─ data_quality_report.md
│  └─ final_report.md
├─ models/
├─ outputs/
├─ experiments/
└─ requirements.txt
```

---

## Luồng gọi hệ thống

### 1. Luồng Data Quality + Chuẩn bị dữ liệu

```
Tải raw từ Roboflow (COCO JSON, raw images không stretch)
  │
  ├─ scripts/data_quality/dedup_images.py
  │    └─ interim/dedup_clusters.json
  │
  ├─ scripts/data_quality/check_annotations.py
  │    └─ interim/annotation_quality_report.md + flagged_images.json
  │
  ├─ [Review thủ công flagged images → fix/loại bỏ]
  │
  ├─ scripts/data_quality/dataset_stats.py
  │    └─ interim/dataset_statistics.json + plots/
  │
  ├─ scripts/data_quality/split_dataset.py
  │    └─ processed/splits/{train,val,test}.json
  │
  ├─ scripts/convert_coco_dataset.py
  │    ├─ processed/detector/*.jsonl
  │    ├─ processed/field_detector/*.jsonl
  │    └─ processed/e2e/*.jsonl
  │
  └─ scripts/crop_fields.py + scripts/generate_pseudo_labels.py
       ├─ interim/cropped_fields/
       └─ processed/ocr/pseudo_labels.jsonl → [review] → reviewed.jsonl
```

### 2. Luồng train

```
train_detector.py
  ├─ Load configs/train_detector.yaml + configs/augmentation.yaml
  ├─ Load splits/train.json
  ├─ Augmentation pipeline (4 nhóm, letterbox resize)
  ├─ Train YOLOv8/11 class "card"
  ├─ Validate trên splits/val.json
  ├─ Log → experiments/
  └─ Save → models/card_detector/
  
  *** Checkpoint: mAP@0.5 ≥ 0.85 → proceed. Nếu chưa đạt → tune augmentation/hyperparams ***

train_field_detector.py
  ├─ Tương tự, class id/name/birth/origin/address/title
  └─ *** Checkpoint: mAP@0.5 ≥ 0.75 → proceed ***

train_ocr.py (khi đã có reviewed.jsonl)
  ├─ OCR-specific augmentation
  ├─ Fine-tune VietOCR / PaddleOCR
  └─ Save → models/ocr/
```

### 3. Luồng inference — Ảnh tĩnh

```
run_infer.py
  1. Đọc ảnh
  2. [Card Detection] → bbox/polygon
  3. [Rectification] → warp (fallback chain: keypoint→contour→hough→bbox crop)
  4. [Orientation Detection] → xoay đúng hướng
  5. [Image Enhancement] → CLAHE + denoise + quality score
  6. [TTA Check] → auto-trigger nếu quality/conf thấp
  7. [Side Classification] → rule-based
  8. [Field Localization] → field detector (primary) + template crop (fallback)
  9. [OCR] → VietOCR + PaddleOCR ensemble
  10. [Parsing] → validate (soft/hard rules) + confidence routing
  11. [Error Recovery] → decision tree xử lý degraded mode
  12. Xuất JSON + visualize intermediate steps
  13. Log processing_time_ms, tta_applied, rectification_method
```

### 4. Luồng inference — Video

```
run_infer_video.py
  1. Đọc video
  2. [Adaptive Sampling] → sample rate theo độ dài video
  3. [Card Detection] → lọc frames có card
  4. [Quality Scoring] → sharpness + brightness + blur
  5. [Frame Selection] → top-K frames
  6. [Full Pipeline] → chạy image pipeline trên K frames
  7. [Ensemble] → per-field max confidence hoặc majority vote
  8. [Edge Case Handling] → quality insufficient / no card / multi-card
  9. Xuất JSON với video_meta
```

### 5. Luồng đánh giá

```
evaluate.py
  ├─ Detection:
  │    ├─ IoU/P/R/F1 cho card (IoU=0.5, 0.75)
  │    ├─ mAP@0.5, mAP@0.5:0.95 cho fields
  │    ├─ Per-class AP + confusion matrix
  │    └─ So sánh: có TTA vs không TTA
  │
  ├─ OCR:
  │    ├─ CER/WER per field type
  │    ├─ Normalized Edit Distance
  │    ├─ Pattern error analysis (dấu VN, số↔chữ, thiếu ký tự)
  │    ├─ So sánh: VietOCR vs PaddleOCR vs ensemble
  │    └─ So sánh: có TTA vs không TTA
  │
  ├─ End-to-end:
  │    ├─ Field exact match + fuzzy match (Levenshtein ≤ 2)
  │    ├─ Full-record exact match
  │    ├─ Breakdown: ảnh tĩnh vs video frame
  │    └─ Breakdown: ảnh dễ vs ảnh khó (theo quality score)
  │
  ├─ Augmentation Ablation:
  │    ├─ No augmentation → baseline
  │    ├─ Light aug (nhóm 1+2) → compare
  │    ├─ Full aug (4 nhóm) → compare
  │    ├─ Full aug + custom CCCD → compare
  │    └─ Bảng + biểu đồ so sánh
  │
  └─ Output:
       ├─ outputs/metrics_summary.json
       ├─ outputs/confusion_matrix.png
       ├─ outputs/error_analysis.md
       └─ outputs/augmentation_ablation.md
```

---

## Phân chia công việc cho 3 người

### Người 1: Data & Detection

Phụ trách:
- Data Quality Pipeline: dedup, kiểm tra annotation, thống kê dataset, split 70/15/15
- Train Card Detector (YOLOv11, 1 class)
- Train Field Detector (YOLOv11, 6 class)
- Benchmark detection metrics (mAP, precision, recall, confusion matrix)
- Viết phần data & detection cho báo cáo

Deliverables:
- `data/processed/splits/` (train/val/test JSON)
- `model/card_detector/best.pt` + `eval_results.json`
- `model/field_detector/best.pt` + `eval_results.json`
- `scripts/data_quality/*`, `scripts/detection/*`
- thống kê dữ liệu + bảng metric detection cho báo cáo

### Người 2: Image Processing & OCR

Phụ trách:
- Module Rectification (perspective warp thẻ về 856×540)
- Module Image Enhancement (CLAHE, denoising)
- Pseudo-label workflow: chạy VietOCR + PaddleOCR, review transcript
- Tích hợp OCR pipeline (ensemble VietOCR + PaddleOCR)
- Benchmark OCR (CER/WER per field)

Deliverables:
- `src/rectification/*`
- `src/enhancement/*`
- `src/ocr/*`
- `data/processed/ocr/` (field crops + reviewed transcripts)
- bảng metric OCR cho báo cáo

### Người 3: Parsing, Integration & Demo

Phụ trách:
- Mapping class → field chuẩn (id_number, full_name, ...)
- Rule validation và hậu xử lý (regex, fuzzy match tỉnh/thành)
- Pipeline end-to-end (nối tất cả các module)
- Evaluation framework (exact match, fuzzy match, e2e metrics)
- Demo app (Gradio/Streamlit)
- Tổng hợp báo cáo cuối

Deliverables:
- `src/parsing/*`
- `src/pipeline/*`
- `src/evaluation/*`
- `demo/*`
- `docs/final_report.md`

---

## Dependency giữa các người & song song hóa

```
Tuần 1:
  Người 1: Data Quality pipeline (dedup, check annotations, stats, split) → output: splits/
  Người 2: Nghiên cứu Rectification + Enhancement, setup PaddleOCR/VietOCR môi trường
  Người 3: Viết parsing rules (soft/hard) + validation logic + error recovery decision tree

Tuần 2:
  Người 1: Train card detector + field detector (dùng splits/ từ tuần 1)
            *** Checkpoint: detection đạt mAP@0.5 ≥ 0.85 (card), ≥ 0.75 (field) ***
  Người 2: Implement Rectification + Enhancement (dùng ảnh test thủ công)
            Crop fields từ bbox GT + chạy pseudo-label + bắt đầu review transcript
  Người 3: Viết pipeline skeleton + evaluation framework + TTA module

Tuần 3:
  Người 1: Hỗ trợ Người 2 review transcript + viết data quality report + ablation detection
  Người 2: Tích hợp OCR ensemble + benchmark CER/WER + hoàn thành reviewed transcripts
  Người 3: Nối pipeline end-to-end + test error recovery + video pipeline

Tuần 4:
  Người 1: Test real-world ảnh + viết phần data/detection cho báo cáo + finalize metrics
  Người 2: Fine-tune VietOCR nếu kịp + so sánh TTA OCR + viết phần OCR cho báo cáo
  Người 3: Demo app (upload ảnh + video + hiển thị intermediate steps) + tổng hợp báo cáo
```

**Bottleneck chính:** Người 1 phải hoàn thành split trong tuần 1 để Người 2 bắt đầu train. Detection phải PASS trước tuần 3 để Người 3 nối pipeline.

---

## Lộ trình xây dựng

### Giai đoạn 1: Data Quality & Detection (~tuần 1-2)

- Tải raw images từ Roboflow (không dùng stretch version).
- Chạy Data Quality pipeline: dedup → check annotations → stats → split.
- Review flagged images, fix/loại bỏ annotation lỗi.
- Implement augmentation pipeline (4 nhóm + custom CCCD aug).
- Train card detector + field detector với letterbox resize.
- **Gate:** Detection phải đạt mAP@0.5 ≥ 0.85 (card) và ≥ 0.75 (field) trước khi tiếp.

### Giai đoạn 2: OCR Dataset & Recognition (~tuần 2-3)

- Crop field từ bbox GT.
- Chạy pseudo-label (VietOCR + PaddleOCR song song).
- Review transcript (ưu tiên: id_number → full_name → date_of_birth → còn lại).
- Benchmark OCR pretrained.
- Implement OCR ensemble.
- Augmentation ablation study.
- Fine-tune VietOCR nếu đủ thời gian.

### Giai đoạn 3: Integration & Robustness (~tuần 3)

- Nối full pipeline: detection → rectify → orientation → enhance → field → OCR → parse.
- Implement error recovery decision tree.
- Implement TTA (detection + OCR).
- Implement video pipeline (adaptive sampling → quality scoring → ensemble).
- Test trên ảnh thực tế chụp từ điện thoại (ngoài test set).

### Giai đoạn 4: Demo & Báo cáo (~tuần 4)

- Demo app (Gradio/Streamlit): upload ảnh + video, hiển thị từng bước trung gian.
- Tổng hợp metrics: detection, OCR, e2e, ablation, TTA comparison.
- Viết báo cáo học thuật: data quality, methodology, kết quả, analysis, limitations.
- Prepare citation và ethical considerations.

---

## Test Plan

### Detection

- Card detection: P, R, F1 trên test split (IoU = 0.5, 0.75)
- Field detection: mAP@0.5, mAP@0.5:0.95, per-class AP
- Confusion matrix
- So sánh: có/không TTA

### OCR

- CER/WER per field type
- Normalized Edit Distance
- Pattern error: top-10 lỗi phổ biến per field
- So sánh: VietOCR vs PaddleOCR vs ensemble
- So sánh: có/không TTA

### End-to-end

- Exact match per field (đặc biệt `id_number` — field quan trọng nhất)
- Fuzzy match (Levenshtein ≤ 2) cho `full_name`, `origin`, `address`
- Full-record exact match
- Breakdown: ảnh dễ (quality ≥ 0.7) vs ảnh khó (quality < 0.5)
- Breakdown: ảnh tĩnh vs video frame

### Augmentation Ablation

| Config | Augmentation | So sánh |
|---|---|---|
| A | Không augmentation | Baseline |
| B | Nhóm 1+2 only | Geometric + Photometric |
| C | Nhóm 1+2+3+4 | + Degradation + Occlusion |
| D | Nhóm 1-4 + custom CCCD (specular, finger, scratch, fade...) | + Domain-specific damage |
| E | Nhóm 1-5 (thêm Background + Multi-card) | + Context diversity |
| F | Tất cả 6 nhóm + Extreme level 10% | Full pipeline |

- So sánh mAP (detection) và CER (OCR) qua 6 configs
- Vẽ biểu đồ improvement % từng bước
- Đặc biệt so sánh Config A vs F trên real-world test images (20-50 ảnh ngoài dataset)

### Tình huống test thực tế

| Nhóm | Tình huống |
|---|---|
| Góc chụp | Thẳng, nghiêng ≤15°, nghiêng >15°, xoay 90°/180° |
| Ánh sáng | Bình thường, tối, lóa, bóng đổ, ánh sáng không đều, đèn vàng |
| Nền | Đơn sắc, bàn làm việc, tay cầm thẻ, nhiều vật xung quanh |
| Chất lượng | Rõ nét, mờ nhẹ, mờ nặng, JPEG nén nặng, resolution thấp |
| Occlusion | Đầy đủ, ngón tay che mép, lóa chip NFC, vật che 1 phần |
| Video | Frame rõ, motion blur, camera rung, tối, lần lượt quay 2 thẻ |
| Đặc biệt | Ảnh photocopy, ảnh chụp màn hình, thẻ cũ bạc màu |

### Real-world validation (ngoài test set)

- Thu thập 20-50 ảnh CCCD thực tế (chụp từ điện thoại nhóm, điều kiện tự nhiên).
- Chạy pipeline, đánh giá manual accuracy.
- Mục đích: kiểm tra generalization gap giữa dataset Roboflow và thực tế.
- **Lưu ý privacy:** xem section Privacy bên dưới.

---

## Privacy & Security

CCCD chứa thông tin cá nhân nhạy cảm. Dù là PoC học thuật, cần tuân thủ:

- **Dataset Roboflow:** dữ liệu public, license CC BY 4.0 — OK để dùng. Tuy nhiên nên lưu ý dataset có thể chứa ảnh CCCD thật với thông tin cá nhân.
- **Real-world test images:** ảnh CCCD của thành viên nhóm hoặc người tình nguyện có đồng ý.
- **Inference:** chạy local, KHÔNG upload ảnh lên cloud service.
- **Output JSON:** chỉ lưu path đến artifacts, không embed ảnh gốc trong JSON.
- **Demo app:** thêm option xóa ảnh và kết quả sau khi xử lý.
- **Báo cáo:** KHÔNG include ảnh CCCD thật có thông tin cá nhân đọc được. Blur/mask thông tin nếu cần minh họa.
- **Git:** KHÔNG commit ảnh CCCD vào repo. Thêm `data/raw/`, `data/interim/cropped_fields/` vào `.gitignore`.

---

## Versioning & Reproducibility

- **Code:** Git, commit message rõ ràng, branch per feature.
- **Data:** lưu MD5 checksum cho file annotation gốc + ghi rõ Roboflow dataset version (v1).
- **Augmentation:** config lưu `configs/augmentation.yaml`, mỗi experiment ghi rõ config nào.
- **Split:** seed cố định (42), lưu file split JSON trong repo.
- **Experiments:** mỗi run ghi lại: config, seed, aug config, #epochs, best metric, checkpoint path → `experiments/experiment_log.csv`.
- **Dependencies:** `requirements.txt` pin version cụ thể.
- **Optional:** MLflow/W&B nếu nhóm có thời gian.

---

## Giả định đã khóa

- V1 ưu tiên `mặt trước CCCD`.
- Dataset Roboflow (cccd by Interlock, v1) là nguồn chính cho detection/localization.
- **Tải raw images, KHÔNG dùng version stretch 640×640.**
- OCR chưa train trực tiếp — dùng pseudo-label workflow.
- `Field detector` là primary; `template crop` là fallback.
- Detection phải đạt mAP gate trước khi chuyển sang OCR.
- `mặt sau`, `mrz`, `issue_date`, `expiry_date`, `sex`, `nationality` chỉ làm khi có thêm data.
- Pipeline V1 xử lý **single card per image/frame**.
- Post-processing rules thiết kế **mềm** (soft warning > hard reject).
- Inference target: < 2s/ảnh (GPU, không TTA), < 6s/ảnh (GPU, có TTA).
- Privacy: inference local, không upload cloud, không commit ảnh vào Git.

---

## Tóm tắt toàn bộ cải tiến so với plan gốc

| # | Cải tiến | Lý do |
|---|---|---|
| 1 | Nguồn dataset cụ thể + xử lý vấn đề stretch/split | Dataset Roboflow bị stretch méo, split không cân |
| 2 | Data Quality Pipeline 5 bước (dedup, annotation check, stats, split) | Tránh train trên data lỗi, ngăn data leakage |
| 3 | Tiêu chí chuyển giao Detection → OCR (mAP gate) | Tránh garbage in garbage out |
| 4 | Chiến lược OCR chi tiết: phân tách field, thứ tự fine-tune, ensemble, retry | Mỗi field cần chiến lược khác nhau |
| 5 | Pseudo-label workflow giảm 60-70% công annotation | Semi-auto hiệu quả hơn manual |
| 6 | Letterbox resize thay vì stretch | Giữ tỷ lệ thẻ, bbox chính xác |
| 7 | Orientation Detection module | Xử lý ảnh xoay 90°/180° |
| 8 | Rectification fallback chain (keypoint→contour→hough→crop) | Plan gốc chỉ có contour, dễ fail |
| 9 | Post-processing rules mềm (soft warning > hard reject) | Tránh loại nhầm edge case |
| 10 | Error Recovery decision tree | Pipeline luôn cho ra kết quả partial thay vì crash |
| 11 | Augmentation 6 nhóm domain-aware + 10 custom CCCD aug + OCR aug riêng | Cover toàn bộ degradation thực tế, không chỉ augmentation generic |
| 12 | Nhóm 5 Context & Background (thay nền, multi-card distraction) | Dataset thường nền sạch, thực tế nền đa dạng — lỗ hổng lớn nhất |
| 13 | Nhóm 4 mở rộng (scratch, local fade, local overexposure) | Thẻ cũ hư hỏng, flash chụp gần — rất phổ biến thực tế |
| 14 | Moiré simulation, grayscale/channel shuffle | Chụp qua màn hình, app scan B&W |
| 15 | Compose 4 level (Light/Medium/Heavy/Extreme) thay vì 3 | Thêm Extreme 10% mô phỏng worst case, compound degradation |
| 16 | OCR augmentation mở rộng (background noise, line through, BrightnessContrast, compression) | OCR crop thực tế không có nền sạch |
| 17 | Tài nguyên thu thập cho custom aug (textures, distraction cards, bg textures) | Custom aug cần assets thực tế |
| 14 | TTA với auto-trigger + time budget | Tăng robustness khi ảnh khó, có kiểm soát thời gian |
| 15 | Video pipeline (adaptive sampling + quality scoring + ensemble + edge cases) | Hỗ trợ video thực tế |
| 16 | Augmentation ablation study | Chứng minh giá trị augmentation cho báo cáo |
| 17 | Real-world validation (20-50 ảnh ngoài dataset) | Kiểm tra generalization gap |
| 18 | Demo hiển thị intermediate steps | Debug + minh họa cho báo cáo |
| 19 | Privacy & Security guidelines | CCCD chứa thông tin nhạy cảm, cần cho ethical review |
| 20 | Citation dataset + license | Yêu cầu bắt buộc cho báo cáo học thuật |
