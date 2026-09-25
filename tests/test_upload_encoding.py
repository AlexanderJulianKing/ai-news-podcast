"""Titles must survive the Pi's Latin-1 locale (its system LANG is en_US)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _latin1_env():
    """An environment whose default text encoding is not UTF-8, or None if this machine has none."""
    for name in ("en_US.ISO8859-1", "en_US"):   # macOS name, then the Pi's
        env = dict(os.environ, LANG=name, LC_ALL=name, LANGUAGE=name, PYTHONUTF8="0")
        env.pop("PYTHONIOENCODING", None)
        probe = subprocess.run([sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
                               env=env, capture_output=True, text=True)
        if probe.returncode == 0 and "utf" not in probe.stdout.lower():
            return env
    return None


def test_titles_read_as_utf8_under_a_non_utf8_locale(tmp_path):
    env = _latin1_env()
    if env is None:
        pytest.skip("no Latin-1 locale available on this machine")
    titles = tmp_path / "titles.txt"
    titles.write_text("What could Iran’s offer mean, Second title", encoding="utf-8")
    code = (
        "from newscaster.upload import read_episode_titles; import sys; "
        f"sys.stdout.buffer.write(read_episode_titles({str(titles)!r}).encode('utf-8'))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, check=True)
    assert out.stdout.decode("utf-8").startswith("What could Iran’s offer mean")

    # The old plain open() garbles the same file under the same locale.
    old = f"import sys; sys.stdout.buffer.write(open({str(titles)!r}).read().encode('utf-8'))"
    garbled = subprocess.run([sys.executable, "-c", old], env=env, capture_output=True, check=True)
    assert "Iran’s" not in garbled.stdout.decode("utf-8")
