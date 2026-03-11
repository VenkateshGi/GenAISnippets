"""
04_multimodal.py — Gemini Vision: analyze images alongside text
Covers: image from URL, image from file, structured image analysis, document parsing
"""
import base64
import requests
from pathlib import Path
import google.generativeai as genai
from config import get_model, GEMINI_FLASH, PRECISE_GEN_CONFIG, SAFETY_SETTINGS


# ── Load Image from URL ────────────────────────────────────────────────────
def image_from_url(url: str) -> genai.types.BlobDict:
    """Download an image and convert to Gemini-compatible blob."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    ext_to_mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    suffix = Path(url.split("?")[0]).suffix.lower()
    mime   = ext_to_mime.get(suffix, "image/jpeg")
    return {"mime_type": mime, "data": response.content}


# ── Load Image from File ───────────────────────────────────────────────────
def image_from_file(path: str) -> genai.types.BlobDict:
    ext_to_mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".webp": "image/webp"}
    p    = Path(path)
    mime = ext_to_mime.get(p.suffix.lower(), "image/jpeg")
    return {"mime_type": mime, "data": p.read_bytes()}


# ── 1. Simple Image Description ────────────────────────────────────────────
def describe_image(image_source: str, is_url: bool = True) -> str:
    model = get_model(GEMINI_FLASH)
    img   = image_from_url(image_source) if is_url else image_from_file(image_source)
    return model.generate_content([img, "Describe this image in detail."]).text


# ── 2. Structured Image Analysis ──────────────────────────────────────────
def analyze_chart(image_source: str, is_url: bool = True) -> str:
    """Extract structured data from a chart or graph image."""
    model  = get_model(GEMINI_FLASH)
    img    = image_from_url(image_source) if is_url else image_from_file(image_source)
    prompt = """
    Analyze this chart/graph and extract:
    1. Chart type
    2. Title and axes labels
    3. Key data points or trends
    4. Main insight or conclusion
    
    Format as a structured JSON response.
    """
    return model.generate_content([img, prompt]).text


# ── 3. Multi-Image Comparison ──────────────────────────────────────────────
def compare_images(url1: str, url2: str, comparison_prompt: str) -> str:
    model  = get_model(GEMINI_FLASH)
    img1   = image_from_url(url1)
    img2   = image_from_url(url2)
    return model.generate_content([
        img1,
        "Image 1 above.",
        img2,
        f"Image 2 above. {comparison_prompt}",
    ]).text


# ── 4. Document / Resume Parsing ───────────────────────────────────────────
def parse_document_image(image_path: str) -> dict:
    """
    Extract structured info from a scanned document or screenshot.
    Works for resumes, invoices, forms, etc.
    """
    import json, re
    model  = genai.GenerativeModel(
        model_name=GEMINI_FLASH,
        generation_config=PRECISE_GEN_CONFIG,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="Extract structured information from document images. Return only valid JSON.",
    )
    img    = image_from_file(image_path)
    prompt = """
    Extract all information from this document and return as JSON with keys:
    - document_type: type of document
    - extracted_fields: {key: value} pairs of all visible fields
    - raw_text: full extracted text
    """
    response = model.generate_content([img, prompt])
    clean    = re.sub(r"```(?:json)?\s*|\s*```", "", response.text).strip()
    try:
        return json.loads(clean)
    except Exception:
        return {"raw_text": response.text}


# ── 5. Video Frame Analysis (Gemini supports video too) ───────────────────
def analyze_video_url(video_url: str, question: str) -> str:
    """
    Gemini 1.5 Pro can process video files up to 1 hour.
    For large files, use the File API (genai.upload_file).
    """
    model    = get_model(GEMINI_FLASH)
    # For video from URL, use Part with video/mp4 mime type
    response = model.generate_content([
        {"mime_type": "video/mp4", "file_uri": video_url},
        question,
    ])
    return response.text


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use a freely available sample image
    sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    print("=== Image Description ===")
    desc = describe_image(sample_image_url)
    print(desc)

    print("\n=== Chart Analysis (using same image as demo) ===")
    analysis = analyze_chart(sample_image_url)
    print(analysis[:500])
