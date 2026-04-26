"""
Debug script to show detailed extraction results
"""

import sys
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import CCCDPipeline

def debug_extraction(image_path: str):
    """Show detailed extraction results for debugging."""
    
    print(f"\n{'='*80}")
    print(f"DEBUGGING EXTRACTION: {image_path}")
    print(f"{'='*80}\n")
    
    # Initialize pipeline
    pipeline = CCCDPipeline(
        card_detector_path="model/card_detector/best.pt",
        field_detector_path="model/field_detector/best.pt"
    )
    
    # Run extraction
    result = pipeline(image_path)
    
    # 1. Processing Steps
    print("📋 PROCESSING STEPS:")
    print("-" * 80)
    for step in result.processing_steps:
        status_icon = "✅" if step["status"] == "success" else "⚠️"
        print(f"{status_icon} {step['step']}: {step['details']}")
    
    # 2. Warnings and Errors
    if result.warnings:
        print("\n⚠️  WARNINGS:")
        print("-" * 80)
        for w in result.warnings:
            print(f"  • {w}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        print("-" * 80)
        for e in result.errors:
            print(f"  • {e}")
    
    # 3. OCR Results
    print("\n🔤 OCR RESULTS (Raw):")
    print("-" * 80)
    for class_name, (text, conf) in result.ocr_results.items():
        print(f"  {class_name:12} | Text: '{text:30}' | Confidence: {conf:.4f}")
    
    # 4. Parsed Results
    print("\n✨ PARSED RESULTS:")
    print("-" * 80)
    if result.parsed_info:
        info = result.parsed_info
        print(f"  ID Number:         {info.id_number}")
        print(f"  Full Name:         {info.full_name}")
        print(f"  Date of Birth:     {info.date_of_birth}")
        print(f"  Place of Origin:   {info.place_of_origin}")
        print(f"  Place of Residence: {info.place_of_residence}")
        
        # Confidence scores
        print("\n  Confidence Scores:")
        for field, conf in info.confidence_scores.items():
            print(f"    {field:20}: {conf}")
        
        # Validation Errors
        if info.validation_errors:
            print("\n  Validation Errors:")
            for err in info.validation_errors:
                print(f"    • {err}")
        
        # Validation Warnings
        if info.validation_warnings:
            print("\n  Validation Warnings:")
            for warn in info.validation_warnings:
                print(f"    • {warn}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Example usage: python debug_extraction.py <image_path>
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_images = list(Path("data/interim/cropped_fields/id").glob("*.jpg"))[:1]
        if not test_images:
            print("❌ No test images found. Usage: python debug_extraction.py <image_path>")
            sys.exit(1)
        image_path = test_images[0]
    
    debug_extraction(str(image_path))
