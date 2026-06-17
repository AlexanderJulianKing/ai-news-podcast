"""Retrieve-then-refine augmentation (gated, fail-safe)."""
import os
from unittest.mock import patch

import newscaster.config as cfg
import newscaster.pipeline as pipeline
from newscaster.rag.store import Hit


def _hit(text):
    return Hit(chunk_id="c", date="2026_03_01", chunk_type="article",
               outlet="NPR", headline="h", url="u", text=text, similarity=0.9)


def test_augment_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", False)
    with patch("newscaster.pipeline.retrieve_prior_research") as mock_ret:
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "draft summary"
    mock_ret.assert_not_called()


def test_augment_no_hits_returns_draft(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", True)
    with patch("newscaster.pipeline.retrieve_prior_research", return_value=[]), \
         patch("newscaster.pipeline.get_llm_response") as mock_llm:
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "draft summary"
    mock_llm.assert_not_called()


def test_augment_with_hits_calls_refine(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", True)
    with patch("newscaster.pipeline.retrieve_prior_research", return_value=[_hit("old context")]), \
         patch("newscaster.pipeline.get_llm_response", return_value="enriched") as mock_llm:
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "enriched"
    user_prompt = mock_llm.call_args[0][0]
    assert "old context" in user_prompt and "draft summary" in user_prompt


def test_augment_failure_falls_back_to_draft(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", True)
    with patch("newscaster.pipeline.retrieve_prior_research", side_effect=RuntimeError("boom")):
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "draft summary"


def test_augment_end_to_end_with_real_store(tmp_path, monkeypatch):
    """Real ResearchIndex + store.search + context-building; only the embed and
    LLM boundaries are mocked. Catches Hit-shape drift between store and pipeline."""
    from newscaster.rag.store import ResearchIndex, Chunk
    monkeypatch.chdir(tmp_path)
    os.makedirs("logs")  # print_and_write writes a daily log here (prod creates it via _ensure_output_dirs)
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", True)

    store = ResearchIndex()  # DEFAULT_DB_PATH is cwd-relative -> under tmp_path
    store.upsert([Chunk(chunk_id="c0", date="2026_03_01", arc_slug=None, slot=0,
                        chunk_type="article", outlet="NPR", headline="h", url="u",
                        text="prior coverage about the dam", vector=[1.0, 0.0])])
    store.close()

    captured = {}

    def fake_llm(user_prompt, system_prompt=None, mode="light"):
        captured["user_prompt"] = user_prompt
        return "enriched text"

    with patch("newscaster.rag.retrieve.embed_texts", return_value=[[1.0, 0.0]]), \
         patch("newscaster.pipeline.get_llm_response", side_effect=fake_llm):
        out = pipeline._augment_with_prior_research("today's dam synthesis", "2026_03_09")

    assert out == "enriched text"
    assert "prior coverage about the dam" in captured["user_prompt"]
    assert "NPR" in captured["user_prompt"]  # outlet flows into the citation
