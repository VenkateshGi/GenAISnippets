"""
05_function_calling.py — Gemini Tool/Function Calling
Covers: tool definition, tool execution loop, parallel tool calls, error handling
Building tools and then using them for answering
"""
import json
import math
import requests
from datetime import datetime
import google.generativeai as genai
from config import GEMINI_FLASH, SAFETY_SETTINGS, DEFAULT_GEN_CONFIG


# ── Tool Definitions ───────────────────────────────────────────────────────
# Gemini uses FunctionDeclaration — equivalent to OpenAI's function_call

tools = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_current_date",
            description="Returns today's date and time.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={},
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="calculate",
            description="Evaluates a mathematical expression safely.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "expression": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Math expression e.g. '2 ** 10 + sqrt(16)'"
                    ),
                },
                required=["expression"],
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="search_jobs",
            description="Searches for job listings by title and location (mock).",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "job_title": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Job title to search for"
                    ),
                    "location": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="City or region"
                    ),
                },
                required=["job_title"],
            ),
        ),
    ]
)


# ── Tool Implementations ────────────────────────────────────────────────────
def get_current_date() -> dict:
    return {"date": datetime.now().strftime("%Y-%m-%d"), "time": datetime.now().strftime("%H:%M:%S")}


def calculate(expression: str) -> dict:
    """Safe math evaluation using math module only."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}


def search_jobs(job_title: str, location: str = "Remote") -> dict:
    """Mock job search — replace with real API (LinkedIn, Indeed, etc.)."""
    mock_jobs = [
        {"title": f"Senior {job_title}", "company": "Experian",  "location": location, "salary": "₹30-45 LPA"},
        {"title": f"Principal {job_title}", "company": "Deloitte", "location": location, "salary": "₹40-60 LPA"},
        {"title": job_title,               "company": "S&P Global", "location": location, "salary": "₹35-50 LPA"},
    ]
    return {"jobs": mock_jobs, "count": len(mock_jobs)}


# ── Tool Dispatcher ─────────────────────────────────────────────────────────
TOOL_MAP = {
    "get_current_date": get_current_date,
    "calculate":        calculate,
    "search_jobs":      search_jobs,
}

def dispatch_tool(function_call) -> str:
    """Execute a tool call and return result as JSON string."""
    name   = function_call.name
    args   = dict(function_call.args)
    fn     = TOOL_MAP.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = fn(**args)
    return json.dumps(result)


# ── Agentic Tool Loop ──────────────────────────────────────────────────────
def run_agent_with_tools(user_query: str, max_turns: int = 5) -> str:
    """
    Full agentic loop:
    1. Send user query
    2. If model calls a tool → execute → send result back
    3. Repeat until model returns final text
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_FLASH,
        tools=[tools],
        generation_config=DEFAULT_GEN_CONFIG,
        safety_settings=SAFETY_SETTINGS,
        system_instruction="You are a helpful assistant. Use the available tools to answer questions accurately.",
    )

    messages = [{"role": "user", "parts": [user_query]}]

    for turn in range(max_turns):
        response = model.generate_content(messages)
        candidate = response.candidates[0]

        # Check if model wants to call a tool
        tool_calls = [
            part.function_call
            for part in candidate.content.parts
            if hasattr(part, "function_call") and part.function_call.name
        ]

        if not tool_calls:
            # Final text response
            return candidate.content.parts[-1].text

        # Execute all tool calls (handles parallel tool use)
        tool_results = []
        for call in tool_calls:
            print(f"  🔧 Tool: {call.name}({dict(call.args)})")
            result_json = dispatch_tool(call)
            tool_results.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=call.name,
                        response={"result": json.loads(result_json)},
                    )
                )
            )

        # Append model turn + tool results to history
        messages.append({"role": "model",   "parts": candidate.content.parts})
        messages.append({"role": "user",    "parts": tool_results})

    return "Max turns reached without final answer."


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "What is today's date?",
        "Calculate 2**10 + sqrt(144) for me.",
        "Find AI Engineer jobs in Hyderabad.",
        "What's the date and also calculate 15 * 7?",   # Parallel tool calls
    ]

    for q in queries:
        print(f"\nQ: {q}")
        answer = run_agent_with_tools(q)
        print(f"A: {answer}")
