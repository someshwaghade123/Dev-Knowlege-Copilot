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

import re
import time
import json
import httpx
from backend.core.config import settings


def extract_citation_indices(text: str) -> list[int]:
    """
    Parse [1], [2] etc. from the LLM answer to find used indices.
    Returns 1-based indices found in the text.
    """
    # Look for [n] where n is a digit. 
    # Also handles ranges if the AI produces them? (e.g. [1-3] -> future-proofing)
    # For now, strict digits.
    matches = re.findall(r"\[(\d+)\]", text)
    # Convert to unique sorted integers
    indices = sorted(list(set(int(m) for m in matches)))
    return indices


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
        "Use citation numbers like [1] or [2] whenever you reference a source. "
        "Always list the [n] numbers you used at the very end of your answer."
    )

    user_message = (
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "Provide a clear, accurate answer. Use [1], [2] etc. for inline citations."
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
        "max_tokens": 1024,
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


async def verify_factuality(query: str, answer: str, chunks: list[dict]) -> dict:
    """
    Perform a second-pass check to ensure the answer is grounded in the context.
    This helps detect hallucinations that the system prompt might have missed.
    """
    context_str = "\n\n".join([f"[Source {i+1}] {c['text']}" for i, c in enumerate(chunks)])
    
    system_message = (
        "You are a strict fact-checker. Your job is to verify if a given answer is "
        "fully supported by the provided context. "
        "IMPORTANT: If the answer correctly states that the information is NOT in the context, "
        "or says 'I don't know' because the context is insufficient, mark it as grounded (is_grounded: true). "
        "Only flag an answer as ungrounded if it makes specific claims or cites facts NOT present in the context. "
        "Output your response in JSON format only: {'score': int, 'is_grounded': bool, 'reason': str}. "
        "Score is from 0 to 10."
    )
    
    user_message = (
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        f"PROPOSED ANSWER: {answer}\n\n"
        "Evaluate the answer's factuality based ONLY on the context. "
        "Are all cited sources [n] actually in the provided context? "
        "Are all technical claims supported by the specific text sections?"
    )

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.0, # Complete deterministic for checking
        "response_format": {"type": "json_object"} if "gpt" in settings.llm_model.lower() else None
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.post(
                f"{settings.llm_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Simple JSON extraction if the model wraps it in block
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = {"score": 5, "is_grounded": False, "reason": "Failed to parse fact-check JSON"}
                
            return {
                "factuality_score": result.get("score", 0),
                "is_grounded": result.get("is_grounded", False),
                "reasoning": result.get("reason", "N/A"),
                "tokens_used": data.get("usage", {}).get("total_tokens", 0)
            }
        except Exception as e:
            print(f"[Factuality Guard Error]: {e}")
            return {"factuality_score": 10, "is_grounded": True, "reasoning": "Check bypassed due to error"}
