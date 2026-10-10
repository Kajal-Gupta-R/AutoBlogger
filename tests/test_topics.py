import json

from backend import nodes


def test_pick_topic_skips_used_topics(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "ARTICLES_DIR", tmp_path)
    used = nodes.TOPICS[:-1]
    entries = [{"title": topic} for topic in used]
    (tmp_path / "index.json").write_text(json.dumps(entries), encoding="utf-8")

    assert nodes.pick_topic({})["topic"] == nodes.TOPICS[-1]


def test_pick_topic_falls_back_when_all_used(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "ARTICLES_DIR", tmp_path)
    entries = [{"title": topic} for topic in nodes.TOPICS]
    (tmp_path / "index.json").write_text(json.dumps(entries), encoding="utf-8")

    assert nodes.pick_topic({})["topic"] in nodes.TOPICS