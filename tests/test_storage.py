import json
from datetime import date

from backend import storage


def read_index(folder):
    return json.loads((folder / "index.json").read_text(encoding="utf-8"))


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


def test_saving_same_topic_twice_does_not_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ARTICLES_DIR", tmp_path)

    storage.save_article({"topic": "Same", "final": "v1"})
    storage.save_article({"topic": "Same", "final": "v2"})

    assert len(read_index(tmp_path)) == 1