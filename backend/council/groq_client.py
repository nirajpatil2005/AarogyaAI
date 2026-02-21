"""
Groq API client for the LLM Council.
Uses asyncio.gather for parallel council member queries.

Model assignments (verified live on Groq as of Feb 2026):
  Member A  — llama-3.3-70b-versatile   (fast 70B general reasoning)
  Member B  — llama-3.1-8b-instant      (ultra-fast 8B, diverse perspective)
  Member C  — qwen/qwen3-32b            (Qwen3 32B — strong diverse reasoning)
  Chairman  — llama-3.3-70b-versatile    (synthesis + final answer)
  Reviewer  — llama-3.1-8b-instant       (fast peer ranking only)
"""
import asyncio
from groq import AsyncGroq
from config import get_settings

settings = get_settings()

# ── Model roster (all verified live on Groq) ──────────────────────────────────
COUNCIL_MODELS = {
    "member_a": "llama-3.3-70b-versatile",   # Fast 70B — general reasoning
    "member_b": "llama-3.1-8b-instant",      # Ultra-fast 8B — diverse perspective
    "member_c": "qwen/qwen3-32b",            # Qwen3 32B — strong diverse reasoning
    "chairman": "llama-3.3-70b-versatile",   # Llama 70B — final synthesis
    "reviewer": "llama-3.1-8b-instant",      # Llama 8B — fast peer ranking only
}

# System prompt shared by all council members
COUNCIL_SYSTEM = (
    "You are a clinical reasoning assistant. The patient case has been de-identified. "
    "Reply ONLY with a valid JSON object — no markdown fences, no text outside JSON. "
    'Keys: "differentials" (list of strings), "next_steps" (list of strings), '
    '"confidence" (float 0-1), "red_flag" (boolean).'
)


async def query_groq(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Query a single Groq model and return the text response."""
    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def query_council_parallel(sanitized_prompt: str) -> dict[str, str]:
    """
    Stage 1 — Divergence: fan-out to ALL council members (A, B, C) in parallel.
    Returns {member_name: raw_response_text}.
    """
    system_msg = {"role": "system", "content": COUNCIL_SYSTEM}
    user_msg   = {"role": "user",   "content": sanitized_prompt}

    tasks = {
        name: query_groq(model, [system_msg, user_msg], max_tokens=512)
        for name, model in COUNCIL_MODELS.items()
        if name in ("member_a", "member_b", "member_c")  # only divergence members
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return {
        name: (r if not isinstance(r, Exception) else
               '{"differentials":[],"next_steps":[],"confidence":0,"red_flag":false}')
        for name, r in zip(tasks.keys(), results)
    }
