"""
03_structured_output.py — Force Gemini to return validated JSON / Pydantic models
Pydantic model is helpful in having structured output for any result
Covers: JSON mode, Pydantic parsing, retry on invalid JSON, schema injection
"""
import json
import re
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from config import get_model, GEMINI_FLASH, PRECISE_GEN_CONFIG
import google.generativeai as genai
from config import GEMINI_API_KEY, SAFETY_SETTINGS


# ── Pydantic Schemas ────────────────────────────────────────────────────────
class JobMatch(BaseModel):
    job_title:          str
    company:            str
    match_score:        float = Field(ge=0.0, le=1.0, description="0.0–1.0 match score")
    matched_skills:     list[str]
    missing_skills:     list[str]
    recommendation:     str
    salary_estimate_usd: Optional[str] = None


class ResumeAnalysis(BaseModel):
    candidate_name:   str
    total_experience: str
    top_skills:       list[str]
    skill_gaps:       list[str]
    strengths:        list[str]
    improvements:     list[str]
    overall_score:    float = Field(ge=0.0, le=10.0)


# ── Helper: Extract JSON from LLM response ────────────────────────────────
def extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    # Remove ```json ... ``` wrappers
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    return json.loads(clean)


# ── Method 1: Prompt Engineering (Schema Injection) ───────────────────────
def structured_output_via_prompt(prompt: str, schema: type[BaseModel], max_retries: int = 3) -> BaseModel:
    """
    Instruct the model to return JSON matching the Pydantic schema.
    Retry on parse failures.
    """
    schema_str = json.dumps(schema.model_json_schema(), indent=2)
    system_prompt = f"""
    You are a JSON extraction assistant. 
    ALWAYS respond with ONLY valid JSON matching this schema — no preamble, no markdown:
    
    {schema_str}
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_FLASH,
        generation_config=PRECISE_GEN_CONFIG,
        safety_settings=SAFETY_SETTINGS,
        system_instruction=system_prompt,
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            data     = extract_json(response.text)
            return schema(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise

    raise RuntimeError("Max retries exceeded")


# ── Method 2: Gemini Native JSON Mode (response_mime_type) ────────────────
def structured_output_native_json(prompt: str) -> dict:
    """
    Use Gemini's built-in JSON response mode.
    Available in gemini-1.5-flash / pro.
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_FLASH,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
        safety_settings=SAFETY_SETTINGS,
    )
    response = model.generate_content(prompt)
    return json.loads(response.text)


# ── Practical Use Cases ────────────────────────────────────────────────────
def analyze_resume(resume_text: str) -> ResumeAnalysis:
    prompt = f"""
    Analyze this resume and provide a structured assessment.
    
    Resume:
    {resume_text}
    """
    return structured_output_via_prompt(prompt, ResumeAnalysis)


def match_job(resume_text: str, job_description: str) -> JobMatch:
    prompt = f"""
    Match this resume against the job description and return a detailed analysis.
    
    Resume: {resume_text}
    
    Job Description: {job_description}
    """
    return structured_output_via_prompt(prompt, JobMatch)


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_resume = """
    Venkatesh G — Senior AI/ML Engineer
    Skills: Python, LangChain, LangGraph, Gemini API, AWS (Lambda, SQS, DynamoDB),
            FastAPI, ChromaDB, FAISS, vLLM, Docker
    Experience: 6 years — Amazon (SDE L5), DNB (Senior AI Engineer)
    Projects: Multi-agent recruiter (LangGraph), RAG pipeline, vLLM on EC2
    """

    sample_jd = """
    Principal AI Engineer — FinTech Company
    Requirements: 5+ years Python, LLM/GenAI experience, RAG pipelines,
    LangChain, AWS, vector databases, FastAPI. Agent frameworks a plus.
    """

    print("=== Resume Analysis ===")
    analysis = analyze_resume(sample_resume)
    print(f"Score: {analysis.overall_score}/10")
    print(f"Top Skills: {analysis.top_skills}")
    print(f"Gaps: {analysis.skill_gaps}")

    print("\n=== Job Match ===")
    match = match_job(sample_resume, sample_jd)
    print(f"Match Score: {match.match_score:.0%}")
    print(f"Matched: {match.matched_skills}")
    print(f"Missing: {match.missing_skills}")
    print(f"Recommendation: {match.recommendation}")

    print("\n=== Native JSON Mode ===")
    result = structured_output_native_json(
        "List 3 top vector databases with pros and cons. Return JSON: {databases: [{name, pros, cons}]}"
    )
    print(json.dumps(result, indent=2))
