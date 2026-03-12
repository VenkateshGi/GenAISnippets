"""
01_basic_chat.py — Text generation & multi-turn conversation
Covers: single-turn generation, system prompts, chat history, token counting
"""
import google.generativeai as genai
from config import get_model, GEMINI_FLASH

# ── 1. Simple Single-Turn Generation ──────────────────────────────────────
def simple_generate(prompt: str) -> str:
    model = get_model(GEMINI_FLASH)
    response = model.generate_content(prompt)
    return response.text


# ── 2. System Prompt + Persona ────────────────────────────────────────────
def generate_with_persona(user_message: str) -> str:
    system_prompt = """
    You are an expert Senior AI/ML Engineer.
    Respond concisely and technically. Use code examples when relevant.
    Always mention trade-offs for architectural decisions.
    """
    model = get_model(GEMINI_FLASH, system_prompt=system_prompt)
    response = model.generate_content(user_message)
    return response.text


# ── 3. Multi-Turn Chat Session ─────────────────────────────────────────────
class GeminiChatBot:
    """Stateful multi-turn chatbot using Gemini."""

    def __init__(self, system_prompt: str = None):
        self.model   = get_model(GEMINI_FLASH, system_prompt=system_prompt)
        self.session = self.model.start_chat(history=[])

    def chat(self, user_message: str) -> str:
        response = self.session.send_message(user_message)
        return response.text

    def get_history(self) -> list[dict]:
        """Return chat history as list of dicts."""
        return [
            {"role": msg.role, "content": msg.parts[0].text}
            for msg in self.session.history
        ]

    def count_tokens(self, text: str) -> int:
        return self.model.count_tokens(text).total_tokens


# ── 4. Token Counting & Cost Estimation ───────────────────────────────────
def estimate_cost(prompt: str, response_text: str) -> dict:
    """
    Rough cost estimate (Gemini 1.5 Flash pricing as of 2025).
    Always verify current pricing at ai.google.dev/pricing
    """
    print(GEMINI_FLASH)
    model = get_model(GEMINI_FLASH)
    input_tokens  = model.count_tokens(prompt).total_tokens
    output_tokens = model.count_tokens(response_text).total_tokens

    # Flash: ~$0.075 per 1M input tokens, $0.30 per 1M output tokens
    input_cost  = (input_tokens  / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.30

    return {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(input_cost + output_cost, 6),
    }


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Single-turn
    print("=== Single Turn ===")
    print(simple_generate("What is RAG in 2 sentences?"))

    # With persona
    print("\n=== With Persona ===")
    print(generate_with_persona("When should I use FAISS vs ChromaDB?"))

    # Multi-turn chat
    print("\n=== Multi-Turn Chat ===")
    bot = GeminiChatBot(system_prompt="You are a helpful AI tutor.")
    print(bot.chat("Explain vector embeddings simply."))
    print(bot.chat("Give me a Python code example using those concepts."))
    print(f"\nHistory length: {len(bot.get_history())} messages")

    # Token counting
    prompt = "Explain transformers in detail"
    resp   = simple_generate(prompt)
    costs  = estimate_cost(prompt, resp)
    print(f"\n=== Token Info ===\n{costs}")
