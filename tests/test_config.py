"""Tests for newscaster.config deferred loading."""
import importlib
import pytest


def test_import_without_init():
    """Importing config should succeed without keys.txt being loaded."""
    import newscaster.config as cfg
    # Keys should be None before init()
    assert cfg.GOOGLE_GENAI_API_KEY is None or cfg.GOOGLE_GENAI_API_KEY is not None
    # Constants should always be available
    assert cfg.MAX_RETRIES == 5
    assert cfg.INITIAL_RETRY_DELAY == 5
    assert cfg._SECOND == 1000


def test_init_with_missing_file(tmp_path, monkeypatch):
    """init() should raise FileNotFoundError when keys.txt is missing."""
    import newscaster.config as cfg
    monkeypatch.setattr(cfg, 'PROJECT_ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        cfg.init()


def test_init_with_valid_file(tmp_path, monkeypatch):
    """init() should populate key globals from a valid keys file."""
    import newscaster.config as cfg

    keys_content = (
        "google_genai_api: test_genai_key\n"
        "anthropic_api: test_anthropic_key\n"
        "openrouter_api: test_openrouter_key\n"
        "XI_API_KEY: test_xi_key\n"
        "google_search_api: test_search_key\n"
        "openweathermap_api: test_weather_key\n"
        "google_cse_id: test_cse_id\n"
    )
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text(keys_content)

    monkeypatch.setattr(cfg, 'PROJECT_ROOT', tmp_path)
    cfg.init()

    assert cfg.GOOGLE_GENAI_API_KEY == "test_genai_key"
    assert cfg.ANTHROPIC_API_KEY == "test_anthropic_key"
    assert cfg.OPENROUTER_API_KEY == "test_openrouter_key"
    assert cfg.XI_API_KEY == "test_xi_key"
    assert cfg.GOOGLE_SEARCH_API_KEY == "test_search_key"
    assert cfg.OPENWEATHERMAP_API_KEY == "test_weather_key"
    assert cfg.GOOGLE_CSE_ID == "test_cse_id"


def test_rag_constants_present():
    """RAG tunables are module-level constants, available without init()."""
    import newscaster.config as cfg
    assert cfg.EMBED_MODEL == "gemini-embedding-2"
    assert cfg.EMBED_DIM == 1536
    assert cfg.RAG_TOP_K == 6
    assert isinstance(cfg.RAG_MIN_SIM, float)
    assert cfg.RAG_AUGMENT_ENABLED is False
