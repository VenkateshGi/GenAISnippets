"""
config.py — Central configuration for Gemini GenAI codebase
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── API Setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your-key")

genai.configure(api_key=GEMINI_API_KEY)

# ── Model Names ────────────────────────────────────────────────────────────
GEMINI_FLASH   = "gemini-2.5-flash"        # Fast, cost-efficient
GEMINI_PRO     = "gemini-2.5-pro"          # Best quality
GEMINI_EMBED   = "models/text-embedding-004"

# ── Generation Defaults ────────────────────────────────────────────────────
DEFAULT_GEN_CONFIG = genai.types.GenerationConfig(
    temperature=0.7,
    top_p=0.95,
    max_output_tokens=2048,
)

PRECISE_GEN_CONFIG = genai.types.GenerationConfig(
    temperature=0.0,        # Deterministic for structured tasks
    max_output_tokens=4096,
)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",      "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

def get_model(model_name: str = GEMINI_FLASH, system_prompt: str = None) -> genai.GenerativeModel:
    """Factory: returns a configured GenerativeModel."""
    kwargs = {
        "model_name":        model_name,
        "generation_config": DEFAULT_GEN_CONFIG,
        "safety_settings":   SAFETY_SETTINGS,
    }
    if system_prompt:
        kwargs["system_instruction"] = system_prompt
    return genai.GenerativeModel(**kwargs)
