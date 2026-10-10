import csv
import json
from datetime import date

from backend import storage


def read_index(folder):
    return json.loads((folder / "index.json").read_text(encoding="utf-8"))


def read_archive(folder):
    with (folder / "archive.csv").open(encoding="utf-8", newline="") as archive_file:
        return list(csv.DictReader(archive_file))


def test_slugify():
    assert storage.slugify("RAG: How it works!") == "rag-how-it-works"


def test_save_article_writes_file_and_index(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ARTICLES_DIR", tmp_path)

    result = storage.save_article({"topic": "My Topic", "final": "# Hello"})

    expected = f"{date.today().isoformat()}-my-topic.md"
    assert result == {"filename": expected}
    assert (tmp_path / expected).read_text(encoding="utf-8") == "# Hello"
    index = read_index(tmp_path)
    assert index[0]["file"] == expected
    assert index[0]["title"] == "My Topic"
    archive = read_archive(tmp_path)
    assert archive == [
        {
            "date": date.today().isoformat(),
            "title": "My Topic",
            "link": f"../#{expected}",
            "content": "# Hello",
        }
    ]


def test_saving_same_topic_twice_does_not_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ARTICLES_DIR", tmp_path)

    storage.save_article({"topic": "Same", "final": "v1"})
    storage.save_article({"topic": "Same", "final": "v2"})

    assert len(read_index(tmp_path)) == 1
    archive = read_archive(tmp_path)
    assert len(archive) == 1
    assert archive[0]["content"] == "v2"


def test_saving_article_archives_existing_articles(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ARTICLES_DIR", tmp_path)
    older_file = "2026-01-01-older.md"
    (tmp_path / older_file).write_text("Older article", encoding="utf-8")
    (tmp_path / "index.json").write_text(
        json.dumps([{"title": "Older", "file": older_file, "date": "2026-01-01"}]),
        encoding="utf-8",
    )

    storage.save_article({"topic": "New", "final": "New article"})

    archive = read_archive(tmp_path)
    assert [row["title"] for row in archive] == ["New", "Older"]
    assert archive[1]["content"] == "Older article"