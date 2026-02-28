"""
backend/generation/llm.py
──────────────────────────
Calls the LLM to generate an answer from retrieved chunks.

THE GENERATION STEP IN RAG:
  RAG = Retrieval-Augmented Generation
             ▲                  ▲
             │                  └── This file handles the "Generation" part
             └─────────────────── vector_store.py handles the "Retrieval" part

  Flow:
    1. User asks: "How do I configure CORS in FastAPI?"
    2. Retriever finds 5 most relevant chunks from the docs
    3. We inject those chunks into a prompt as "context"
    4. LLM reads the context + question and writes an answer
    5. We return the answer + cite which chunks it came from

WHY NOT JUST ASK THE LLM DIRECTLY?
  Without retrieval: LLM might hallucinate or give outdated info.
  With retrieval (RAG): LLM is grounded in your actual documents.

PROMPT DESIGN:
  We use a "stuffing" strategy — all chunks go into one prompt.
  For Week 1 this is fine. Later we can explore "map-reduce" for very long
  context.

INTERVIEW TIP:
  "The prompt includes a SYSTEM message instructing the model to ONLY answer
   from the provided context and say 'I don't know' if the answer isn't there.
   This is the simplest hallucination guard — we measure how often it's
   triggered to track hallucination rate."
"""

import time
import httpx
from backend.core.config import settings


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for a chat-completion API call.

    Args:
        query:  The user's question
        chunks: List of chunk dicts with 'text', 'title', 'source_url' fields

    Returns:
        List of {"role": ..., "content": ...} dicts (OpenAI chat format)
    """
    # Format each chunk as a numbered context block
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown")
        url = chunk.get("source_url", "")
        text = chunk.get("text", "")
        context_blocks.append(
            f"[Context {i}] Source: {title}\nURL: {url}\n\n{text}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    system_message = (
        "You are a precise technical documentation assistant. "
        "Answer the user's question using ONLY the provided context blocks. "
        "If the answer is not in the context, say: "
        "'I could not find a reliable answer in the indexed documents.' "
        "Always cite which Context number(s) you used at the end of your answer."
    )

    user_message = (
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "Provide a clear, accurate answer with citation numbers."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


async def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Call the LLM API and return the answer with token + latency metadata.

    Args:
        query:  The user's question
        chunks: Retrieved chunks from vector_store.search()

    Returns:
        {
          "answer": str,
          "tokens_used": int,     ← prompt + completion tokens
          "generation_latency_ms": int
        }
    """
    messages = build_prompt(query, chunks)

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,   # Low temperature → more factual, less creative
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{settings.llm_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            if response.status_code != 200:
                print(f"[LLM API Error] Status {response.status_code}: {response.text}")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise Exception(f"LLM API Error: {e.response.status_code}")
        except Exception as e:
            print(f"[LLM Unexpected Error]: {str(e)}")
            raise

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    data = response.json()

    if "choices" not in data or not data["choices"]:
        print(f"[LLM Error] Invalid response shape: {data}")
        raise Exception("LLM returned an empty or invalid response.")

    answer = data["choices"][0]["message"]["content"]
    tokens_used = data.get("usage", {}).get("total_tokens", 0)

    return {
        "answer": answer,
        "tokens_used": tokens_used,
        "generation_latency_ms": elapsed_ms,
    }
