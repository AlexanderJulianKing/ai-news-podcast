"""Tests for size-based rotation of the JSONL audit logs."""
import json

import newscaster.logging as L


def test_write_jsonl_log_no_date_in_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    L.write_jsonl_log("audit_test", {"event": "x"})
    path = tmp_path / "logs" / "audit_test.jsonl"
    assert path.exists()
    # The date is gone from the filename, but each record carries a timestamp so we
    # don't lose "when" a line was written.
    record = json.loads(path.read_text().splitlines()[0])
    assert "timestamp" in record


def test_write_jsonl_log_rotates_by_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(L, "LOG_MAX_BYTES", 200)
    monkeypatch.setattr(L, "LOG_BACKUP_COUNT", 2)

    for i in range(60):
        L.write_jsonl_log("audit_test", {"i": i, "pad": "x" * 60})

    logs = tmp_path / "logs"
    base = logs / "audit_test.jsonl"
    assert base.exists()
    # Rotation produced backups, bounded by LOG_BACKUP_COUNT (no .3 when count is 2).
    assert (logs / "audit_test.jsonl.1").exists()
    assert (logs / "audit_test.jsonl.2").exists()
    assert not (logs / "audit_test.jsonl.3").exists()
    # The active file stays under the cap and holds the most recent record.
    assert base.stat().st_size <= L.LOG_MAX_BYTES
    assert '"i": 59' in base.read_text().splitlines()[-1]
