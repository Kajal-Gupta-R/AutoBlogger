import csv
import json
import re
from datetime import date
from pathlib import Path

from backend.state import BlogState

ARTICLES_DIR = Path("frontend/articles")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _write_archive(index: list[dict]) -> None:
    with (ARTICLES_DIR / "archive.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archive_file:
        writer = csv.DictWriter(
            archive_file, fieldnames=["date", "title", "link", "content"]
        )
        writer.writeheader()
        for entry in index:
            writer.writerow(
                {
                    "date": entry["date"],
                    "title": entry["title"],
                    "link": f"../#{entry['file']}",
                    "content": (ARTICLES_DIR / entry["file"]).read_text(
                        encoding="utf-8"
                    ),
                }
            )


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
    _write_archive(index)

    return {"filename": filename}
