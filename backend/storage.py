import json
import re
from datetime import date
from pathlib import Path

from backend.state import BlogState

ARTICLES_DIR = Path("frontend/articles")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_article(state: BlogState) -> dict:
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    filename = f"{today}-{slugify(state['topic'])}.md"
    (ARTICLES_DIR / filename).write_text(state["final"], encoding="utf-8")

    index_path = ARTICLES_DIR / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = []

    index = [entry for entry in index if entry["file"] != filename]
    index.insert(0, {"title": state["topic"], "file": filename, "date": today})
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    return {"filename": filename}
