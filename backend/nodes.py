import random

from langchain_groq import ChatGroq

from backend.config import settings
from backend.state import BlogState

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.model_name,
    temperature=0.7,
)

TOPICS = [
    "How retrieval-augmented generation (RAG) works",
    "Transformers explained from first principles",
    "Fine-tuning vs prompt engineering: when to use which",
    "Vector databases and how similarity search works",
    "Evaluating LLM applications in production",
    "How LangGraph models agent workflows as graphs",
    "Embeddings: turning text into numbers that capture meaning",
    "Common failure modes of LLM agents and how to fix them",
]


def ask(prompt: str) -> str:
    """Send one prompt to the LLM and return the text reply."""
    return llm.invoke(prompt).content


def pick_topic(state: BlogState) -> dict:
    return {"topic": random.choice(TOPICS)}


def write_outline(state: BlogState) -> dict:
    prompt = (
        f"Create a clear section-by-section outline for a technical blog "
        f"article titled: {state['topic']}.\n"
        "Include an introduction, 4 to 5 main sections, and a conclusion. "
        "Return only the outline in Markdown."
    )
    return {"outline": ask(prompt)}


def write_draft(state: BlogState) -> dict:
    prompt = (
        f"Write a technical blog article titled: {state['topic']}.\n"
        f"Follow this outline exactly:\n{state['outline']}\n\n"
        "Rules: clear and practical, no filler, use Markdown headings, "
        "include a short code example where it helps. "
        "Return only the article."
    )
    return {"draft": ask(prompt)}


def review(state: BlogState) -> dict:
    prompt = (
        "You are a careful technical editor. Improve this article: fix "
        "errors, tighten wording, and make sure the structure flows. "
        "Keep the Markdown format. Return only the final article.\n\n"
        f"{state['draft']}"
    )
    return {"final": ask(prompt)}
