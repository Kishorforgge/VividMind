"""Ask the LLM to answer using only the retrieved passages (streamed)."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# If the chosen model is retired, try these in order (Google renames models often).
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
NOT_FOUND = "I couldn't find this in the uploaded documents."

PROMPT = """You are VividMind, an assistant that answers questions using ONLY the context passages below.

Rules:
- Use only the context. Never use outside knowledge.
- Cite the source after each fact, like [filename p.3].
- Be concise. Use short bullet points when listing things.
- The context may be a sample spread across the document. For summary or "main character" style questions, answer from what the passages show.
- If the answer is not in the context, reply exactly: "I couldn't find this in the uploaded documents."
- Use the conversation history only to understand follow-up questions.

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer:"""


def _history_text(history):
    lines = []
    for m in (history or [])[-6:]:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:500]}")
    return "\n".join(lines) or "(none)"


def _is_missing_model(err) -> bool:
    s = str(err)
    return "404" in s or "NOT_FOUND" in s or "no longer available" in s


def stream_answer(question: str, hits: list, history=None, api_key=None):
    """Yield the answer in pieces so the UI can show it as it is written."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        yield "Add your GEMINI_API_KEY to the .env file (see .env.example), then restart the app."
        return

    context = "\n\n".join(f"[{h['file']} p.{h['page']}]\n{h['text']}" for h in hits)
    prompt = PROMPT.format(
        history=_history_text(history), context=context, question=question
    )
    client = genai.Client(api_key=key)

    models = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]
    last_error = None
    for model in models:
        got_text = False
        try:
            for chunk in client.models.generate_content_stream(model=model, contents=prompt):
                if chunk.text:
                    got_text = True
                    yield chunk.text
            return
        except Exception as e:
            if got_text or not _is_missing_model(e):
                yield f"\n\nSomething went wrong calling the model ({model}): {e}"
                return
            last_error = e  # model retired: try the next one

    yield (
        "None of the models I tried are available: " + ", ".join(models) + ". "
        "Open aistudio.google.com, find a current Flash model name and set it as "
        f"GEMINI_MODEL in your .env file. Last error: {last_error}"
    )
