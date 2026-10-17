"""The fact check sees which outlet and page each piece of evidence came from (2026-10-17).

Before this, research excerpts reached the fact check as bare page text. On 2026-10-13 the
script aired "a September ninth DefenseScoop report"; the right date (Oct 9) was only in
DefenseScoop's URL, so neither the checker nor the editor could see it. Misattributions
(CNBC's figure credited to Al Jazeera) were invisible for the same reason.
"""
import json

from newscaster import review


def _write_story(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "segment_summaries"
    d.mkdir()
    (d / "2026_10_13_segment1_article0_source.txt").write_text("The Pentagon revised its count.", encoding="utf-8")
    research = {
        "articles": [{"chunk_id": "2026_10_13_seg1_art0", "outlet": "ABC News",
                      "url": "https://abcnews.go.com/US/pentagon-count", "published_date": "2026-10-12"}],
        "followups": [{"source_hunter_sources": [
            {"title": "Pentagon moving to modify its casualty portal | DefenseScoop",
             "url": "https://defensescoop.com/2026/10/09/dcas-changes/",
             "excerpt": "The official said it was triggered by an internal audit of our reporting systems."}]}],
    }
    (d / "2026_10_13_segment1_research.json").write_text(json.dumps(research), encoding="utf-8")


def test_segment_corpus_labels_each_source_with_its_outlet_and_link(tmp_path, monkeypatch):
    _write_story(tmp_path, monkeypatch)
    corpus = review.build_segment_corpus("2026_10_13", 1)
    assert "SOURCE: ABC News — https://abcnews.go.com/US/pentagon-count (published 2026-10-12)" in corpus
    assert "SOURCE: Pentagon moving to modify its casualty portal | DefenseScoop — https://defensescoop.com/2026/10/09/dcas-changes/" in corpus
    assert corpus.index("SOURCE: Pentagon moving") < corpus.index("internal audit of our reporting")


def test_every_chunk_of_a_long_source_keeps_its_label():
    header = "SOURCE: CNBC — https://www.cnbc.com/2026/10/14/hormuz.html"
    chunks = review._source_chunks(header + "\n" + "Exports fell sharply. " * 600, chunk_chars=3500, overlap=350)
    assert len(chunks) > 1 and all(c.startswith(header) for c in chunks)


def test_editor_gets_the_whole_story_evidence_when_it_fits():
    # 2026-10-17: the snippet picker dropped the decisive passage twice (AARP's own "80 percent";
    # the DefenseScoop report), so the editor sees everything up to its 140,000-character limit.
    corpus = "SOURCE: AARP — https://www.aarp.org/x\n80 percent said 3.5 percent would not be enough.\n\n" + "filler. " * 5000
    assert review._editor_evidence("FLAG: eighty percent — CNBC says 76%", corpus) == corpus
    huge = corpus * 10
    picked = review._editor_evidence("FLAG: eighty percent — CNBC says 76%", huge)
    assert 0 < len(picked) <= review._EDITOR_SNIPPET_CHARS + 200
