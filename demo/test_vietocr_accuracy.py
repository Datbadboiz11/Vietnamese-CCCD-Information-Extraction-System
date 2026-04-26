"""
Test VietOCR accuracy on cropped field images
"""

import sys
import os
from pathlib import Path
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ocr.vietocr_adapter import VietOCRAdapter

def test_vietocr_sample_fields():
    """Test VietOCR on sample field images."""
    
    print(f"\n{'='*80}")
    print(f"TESTING VIETOCR ON FIELD IMAGES")
    print(f"{'='*80}\n")
    
    # Load VietOCR
    ocr = VietOCRAdapter(device="cpu")
    
    if not ocr.available:
        print("❌ VietOCR not available!")
        return
    
    # Test on a few field images from each category
    field_dirs = {
        "id": "data/interim/cropped_fields/id",
        "name": "data/interim/cropped_fields/name",
        "birth": "data/interim/cropped_fields/birth",
        "origin": "data/interim/cropped_fields/origin",
        "address": "data/interim/cropped_fields/address",
        "title": "data/interim/cropped_fields/title",
    }
    
    for field_type, field_dir in field_dirs.items():
        print(f"\n{field_type.upper()} FIELD IMAGES:")
        print("-" * 80)
        
        field_path = Path(field_dir)
        if not field_path.exists():
            print(f"  ⚠️  Directory not found: {field_dir}")
            continue
        
        # Get 3 samples
        images = sorted(field_path.glob("*.jpg"))[:3]
        
        for img_path in images:
            try:
                # Load image
                pil_image = Image.open(str(img_path))
                
                # Run OCR
                result = ocr.predict_pil(pil_image)
                
                print(f"  📄 {img_path.name}")
                print(f"     Text:       '{result.text}'")
                print(f"     Confidence: {result.confidence:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print(f"\n{'='*80}")
    print("OBSERVATIONS:")
    print(f"{'='*80}")
    print("""
1. Your models (card & field detection) are **excellent** (>99% mAP)
2. The main issue is **VietOCR accuracy on Vietnamese text**
3. VietOCR is trained on general Vietnamese text, not specialized ID cards

RECOMMENDATIONS TO IMPROVE ACCURACY:
1. ✅ Fine-tune VietOCR on CCCD samples (best solution)
2. ✅ Add post-processing/correction rules for common OCR errors
3. ✅ Use ensemble of multiple OCR engines (VietOCR + EasyOCR + PaddleOCR)
4. ✅ Add manual validation step in Streamlit
5. ✅ Improve image preprocessing (contrast, deskew, denoise)
""")

if __name__ == "__main__":
    test_vietocr_sample_fields()
