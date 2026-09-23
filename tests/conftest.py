"""Shared test setup."""
import os
import tempfile

# Keep every log write made during tests out of the real logs/ folder. Before this,
# test runs on the Pi wrote fake timeouts and "test prompt" calls into the production
# llm_audit.jsonl (seen 2026-09-15 at 15:40 and 18:54).
os.environ.setdefault("NEWSCASTER_LOG_DIR", tempfile.mkdtemp(prefix="newscaster_test_logs_"))
