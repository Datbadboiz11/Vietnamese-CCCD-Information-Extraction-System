"""
Streamlit Demo App for CCCD Information Extraction
Supports: Upload image → Card detection → Field detection → OCR → Parsing → Display result

Chạy:
  streamlit run demo/app.py

Hoặc từ terminal:
  python -m streamlit run demo/app.py
"""

import json
import tempfile
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
try:
    from src.pipeline import CCCDPipeline
    from src.evaluation import OCREvaluator
    from src.parsing import ConfidenceRouter
    from src.parsing.validators import CCCDParser
except ImportError as e:
    st.error(f"Cannot import src modules: {e}\n\nMake sure you're running from project root.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions for Visualization
# ──────────────────────────────────────────────────────────────────────────────

def _draw_bbox(image: np.ndarray, bbox: list, label: str = "", color: tuple = (0, 255, 0), thickness: int = 2):
    """Draw bounding box on image."""
    if bbox is None or len(bbox) != 4:
        return image
    
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
        cv2.rectangle(image, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), font, 0.6, (255, 255, 255), 1)
    
    return image

def _display_detection_results(original_image: np.ndarray, result):
    """Display visualization of detection results."""
    # Original image with card bbox
    if result.card_detected and result.card_bbox:
        st.markdown("### Original Image with Card Detection")
        viz_image = original_image.copy()
        viz_image = _draw_bbox(viz_image, result.card_bbox, label="CARD", color=(0, 255, 0), thickness=3)
        
        # Convert BGR to RGB for display
        viz_image_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
        st.image(viz_image_rgb, use_container_width=True)

def _display_extraction_table(parsed_info, ocr_results=None):
    """Display extracted information as a formatted table."""
    if not parsed_info:
        return

    # Prepare table data
    table_data = [
        ["Field", "Value"],
        ["id", parsed_info.id_number or "_(not extracted)_"],
        ["name", parsed_info.full_name or "_(not extracted)_"],
        ["birth", parsed_info.date_of_birth or "_(not extracted)_"],
        ["origin", parsed_info.place_of_origin or "_(not extracted)_"],
        ["address", parsed_info.place_of_residence or "_(not extracted)_"],
    ]
    
    # Create HTML table with better styling
    html_table = '<table style="width: 100%; border-collapse: collapse; margin: 10px 0;">'
    
    # Header row
    html_table += '<tr style="background-color: #1f77b4; color: white;">'
    for header in table_data[0]:
        html_table += f'<th style="padding: 12px; text-align: left; border: 1px solid #ddd; font-weight: bold;">{header}</th>'
    html_table += '</tr>'
    
    # Data rows
    for i, row in enumerate(table_data[1:]):
        bg_color = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        html_table += f'<tr style="background-color: {bg_color};">'
        for j, cell in enumerate(row):
            if j == 0:
                html_table += f'<td style="padding: 12px; border: 1px solid #ddd; font-weight: bold; color: #1f77b4;">{cell}</td>'
            else:
                html_table += f'<td style="padding: 12px; border: 1px solid #ddd;">{cell}</td>'
        html_table += '</tr>'
    
    html_table += '</table>'
    
    st.markdown(html_table, unsafe_allow_html=True)


@st.cache_data
def load_jsonl_cache(path: str) -> dict:
    """Load pseudo_labels JSONL → {basename(source_image): {class: (best_text, best_conf)}}

    Key is the bare filename (e.g. 'image127_jpg.rf....jpg') so it matches
    the name of the uploaded file directly.
    """
    import os as _os
    cache: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                src = obj.get("source_image", "")
                cls = obj.get("class", "")
                if src and cls:
                    # Normalize: use only the filename, strip directory prefix
                    key = _os.path.basename(src.replace("\\", "/"))
                    if key not in cache:
                        cache[key] = {}
                    cache[key][cls] = (
                        obj.get("best_text", ""),
                        float(obj.get("best_conf", 0.0)),
                    )
    except FileNotFoundError:
        pass
    return cache







st.set_page_config(
    page_title="CCCD Information Extraction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Vietnamese CCCD Information Extraction System")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (hardcoded)
# ──────────────────────────────────────────────────────────────────────────────

card_detector_path = "model/card_detector/best.pt"
field_detector_path = "model/field_detector/best.pt"
device = "cpu"
card_conf_threshold = 0.5
field_conf_threshold = 0.3
low_conf_threshold = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Initialize Session State
# ──────────────────────────────────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.pipeline_initialized = False

if "results" not in st.session_state:
    st.session_state.results = None

if "cache_hit" not in st.session_state:
    st.session_state.cache_hit = False

if "source" not in st.session_state:
    st.session_state.source = None

if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None


# ──────────────────────────────────────────────────────────────────────────────
# Load Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _paddle_available() -> bool:
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource
def load_pipeline(card_path, field_path, device):
    """Load pipeline (cached). Bật ensemble nếu PaddleOCR đã cài."""
    try:
        use_ensemble = _paddle_available()
        pipeline = CCCDPipeline(
            card_detector_path=card_path,
            field_detector_path=field_path,
            device=device,
            use_ensemble=use_ensemble,
        )
        return pipeline, None, use_ensemble
    except Exception as e:
        return None, f"Error loading pipeline: {e}", False


pipeline, error, _using_ensemble = load_pipeline(card_detector_path, field_detector_path, device)

if error:
    st.error(f"{error}")
    st.info("Make sure model files exist at the paths specified in the sidebar.")
    st.stop()

if _using_ensemble:
    st.sidebar.success("OCR Engine: VietOCR + PaddleOCR ensemble")
else:
    st.sidebar.info("OCR Engine: VietOCR  (cài paddleocr để dùng ensemble)")

# Load JSONL lookup cache (pre-computed ensemble results for dataset images)
_jsonl_cache_path = os.path.join(project_root, "pseudo_labels.jsonl")
ocr_cache = load_jsonl_cache(_jsonl_cache_path)

# Image input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a CCCD image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a photo of Vietnamese CCCD"
    )

with col2:
    st.subheader("Sample Image URL")
    sample_url = st.text_input(
        "Or paste image URL",
        value="",
        help="Leave empty if uploading file"
    )

# Process button
if st.button("Extract Information", use_container_width=True, type="primary"):
    if not uploaded_file and not sample_url:
        st.warning("Please upload an image or provide a URL")
    else:
        with st.spinner("Processing... Please wait"):
            try:
                if uploaded_file:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    temp_file.write(uploaded_file.getbuffer())
                    temp_file.flush()  # Flush to ensure file is written
                    temp_file.close()  # Close before using
                    image_path = temp_file.name
                    st.session_state.source = "upload"
                else:
                    import requests
                    response = requests.get(sample_url)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    temp_file.write(response.content)
                    temp_file.flush()  # Flush to ensure file is written
                    temp_file.close()  # Close before using
                    image_path = temp_file.name
                    st.session_state.source = "url"

                st.session_state.last_image_path = image_path
                result = pipeline(image_path)

                # Override OCR results with pre-computed cache if available
                uploaded_name = uploaded_file.name if uploaded_file else ""
                if uploaded_name and uploaded_name in ocr_cache:
                    cached_fields = ocr_cache[uploaded_name]
                    result.ocr_results.update(cached_fields)
                    # Re-parse with cached values
                    parser = CCCDParser()
                    ocr_list = [
                        {"class": cls, "text": text, "confidence": conf}
                        for cls, (text, conf) in result.ocr_results.items()
                    ]
                    result.parsed_info = parser.parse_batch(ocr_list)
                    st.session_state.cache_hit = True
                else:
                    st.session_state.cache_hit = False

                st.session_state.results = result
                st.success("Processing complete!")

            except Exception as e:
                st.error(f"Error: {e}")

# Display results
if st.session_state.results:
    result = st.session_state.results

    st.divider()
    st.subheader("Extraction Results")

    if st.session_state.get("cache_hit"):
        st.info("Results from cache (ensemble VietOCR + PaddleOCR from dataset)")
    else:
        st.info("Live OCR (VietOCR)")

    col_img, col_table = st.columns(2)

    with col_img:
        if hasattr(st.session_state, 'last_image_path') and st.session_state.last_image_path:
            try:
                original_image = cv2.imread(st.session_state.last_image_path)
                if original_image is not None:
                    _display_detection_results(original_image, result)
            except Exception as e:
                st.warning(f"Could not display visualization: {e}")

    with col_table:
        if result.parsed_info:
            st.markdown("### Extracted Information")
            _display_extraction_table(result.parsed_info, result.ocr_results)

    st.divider()
    
    # OCR Metrics Tab
    st.subheader("OCR Metrics & Confidence Scores")
    
    if result.ocr_results:
        # Create confidence table
        metrics_data = []
        for class_name, (text, confidence) in sorted(result.ocr_results.items()):
            # Confidence threshold warning
            if confidence < 0.6:
                status = " Low"
            elif confidence < 0.8:
                status = " Medium"
            else:
                status = " High"
                
            metrics_data.append({
                "Field": class_name.upper(),
                "Confidence": f"{confidence:.1%}",
                "Status": status,
                "Text": text[:50] + "..." if len(text) > 50 else text,
            })
        
        # Display as table
        import pandas as pd
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        confidences = [c for _, c in result.ocr_results.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        min_conf = min(confidences) if confidences else 0
        max_conf = max(confidences) if confidences else 0
        
        with col1:
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col2:
            st.metric("Min Confidence", f"{min_conf:.1%}")
        with col3:
            st.metric("Max Confidence", f"{max_conf:.1%}")
        with col4:
            low_conf_fields = sum(1 for c in confidences if c < 0.6)
            st.metric("Low Confidence Fields", low_conf_fields)


# ──────────────────────────────────────────────────────────────────────────────
# OCR Performance Metrics Section
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("OCR Performance & Confidence Metrics")

st.markdown("""
This section shows overall OCR performance metrics based on analysis of:
- **Confidence Analysis**: ~22,000 OCR results from `pseudo_labels.jsonl` (ensemble results)
- **Accuracy Analysis**: Ground truth validation from `reviewed.jsonl`
""")
    
st.divider()

# Load and display metrics
report_path = os.path.join(project_root, "ocr_performance_report.json")

if os.path.exists(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    # Confidence Analysis
    st.markdown("### Confidence Distribution Analysis")
    
    
    conf_analysis = report_data.get('confidence_analysis', {})
    
    if conf_analysis:
        # Summary table
        summary_rows = []
        for field in sorted(conf_analysis.keys()):
            field_data = conf_analysis[field]
            if 'overall' in field_data:
                overall = field_data['overall']
                summary_rows.append({
                    'Field': field.upper(),
                    'Samples': f"{field_data['total_samples']:,}",
                    'Avg Confidence': overall['mean_conf'],
                    'Median': overall['median_conf'],
                    'High Conf %': overall['high_conf_pct'],
                    'Low Conf %': overall['low_conf_pct'],
                })
        
        if summary_rows:
            import pandas as pd
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Interpretation
        st.markdown("""
**Interpretation:**
- **ID** (93.78%): Highest confidence - very reliable 
- **Birth** (94.14%): Second highest - very reliable 
- **Name** (90.44%): High confidence - reliable 
- **Origin** (84.63%): Moderate confidence - acceptable 
- **Address** (83.26%): Lower confidence - needs review for complex cases 

**Confidence Tiers:**
-  **High** (≥80%): Likely accurate
-  **Medium** (60-80%): Should verify
- **Low** (<60%): Requires manual review
        """)
    
    st.divider()
    
    # Quality Analysis  
    st.markdown("### Accuracy Analysis vs Ground Truth")
    st.markdown("*(Based on 3,114 validated samples)*")
    
    quality_analysis = report_data.get('quality_analysis', {})
    
    if quality_analysis:
        accuracy_rows = []
        for field in sorted(quality_analysis.keys()):
            field_data = quality_analysis[field]
            if 'overall' in field_data:
                overall = field_data['overall']
                accuracy_rows.append({
                    'Field': field.upper(),
                    'Samples': f"{overall['samples']:,}",
                    'Accuracy': overall['accuracy'],
                    'Correct': overall['correct_samples'],
                    'Errors': overall['error_samples'],
                })
        
        if accuracy_rows:
            import pandas as pd
            df = pd.DataFrame(accuracy_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Export button
    st.markdown("### Download Full Report")
    with open(report_path, 'r', encoding='utf-8') as f:
        report_json = f.read()
    
    st.download_button(
        label="Download OCR Performance Report (JSON)",
        data=report_json,
        file_name="ocr_performance_report.json",
        mime="application/json"
    )

else:
    st.warning("Performance report not generated yet. Run `python scripts/ocr_performance_report.py` to generate.")
    st.info("To generate the report, run in terminal:\n```bash\npython scripts/ocr_performance_report.py\n```")
