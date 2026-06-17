"""Retrieve-then-refine augmentation (gated, fail-safe)."""
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
