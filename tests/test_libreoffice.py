from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from grinning_cat_plus.parsers import libreoffice
from tests.conftest import pid_alive


def _src(workdir: Path, name: str = "input.odp", data: bytes = b"ODP") -> Path:
    path = workdir / name
    path.write_bytes(data)
    return path


def test_binary_prefers_libreoffice_then_soffice(fake_soffice, tmp_path, monkeypatch):
    assert libreoffice.libreoffice_binary().endswith("/soffice")
    other = tmp_path / "otherbin"
    other.mkdir()
    lo = other / "libreoffice"
    lo.write_text("#!/bin/sh\n")
    lo.chmod(0o755)
    monkeypatch.setenv("PATH", f"{other}{os.pathsep}{os.environ['PATH']}")
    assert libreoffice.libreoffice_binary() == str(lo)


def test_missing_binary_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="LibreOffice"):
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path)


def test_success_returns_converted_file_and_uses_isolated_profile(fake_soffice, tmp_path):
    out = libreoffice.convert(_src(tmp_path), "pptx", tmp_path)
    assert out.read_bytes() == b"CONVERTED:ODP"
    assert out.parent.parent == tmp_path  # everything stays inside the caller's workdir
    [call] = fake_soffice.calls()
    assert "--headless" in call
    assert f"-env:UserInstallation={(tmp_path / 'profile').as_uri()}" in call


def test_nonzero_exit_raises_with_output(fake_soffice, tmp_path):
    fake_soffice.mode("fail")
    with pytest.raises(RuntimeError, match=r"exit 3.*boom"):
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path)


def test_non_utf8_output_does_not_mask_the_failure(fake_soffice, tmp_path):
    fake_soffice.mode("badbytes")
    with pytest.raises(RuntimeError, match="exit 5"):
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path)


def test_exit_zero_without_output_reports_libreoffice_output(fake_soffice, tmp_path):
    """Real soffice exits 0 on unreadable input: its own message is the only useful diagnostic."""
    fake_soffice.mode("nooutput")
    with pytest.raises(RuntimeError, match=r"slides\.odp.*produced no pptx.*could not be loaded"):
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path, display_name="slides.odp")


def test_messages_name_the_user_file_not_the_internal_one(fake_soffice, tmp_path):
    fake_soffice.mode("fail")
    with pytest.raises(RuntimeError) as exc:
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path, display_name="Relazione è.odp")
    assert "Relazione è.odp" in str(exc.value)
    assert "input.odp" not in str(exc.value)


def test_timeout_kills_the_whole_process_group(fake_soffice, tmp_path):
    fake_soffice.mode("hang")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path, timeout=1)
    assert time.monotonic() - started < 10
    grandchild = fake_soffice.grandchild_pid()
    deadline = time.monotonic() + 5
    while pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(grandchild)


def test_descendant_outside_the_group_does_not_hang_the_timeout(fake_soffice, tmp_path, monkeypatch):
    """A descendant that left the killed process group still holds the pipes:
    the bounded wait must stay bounded."""
    fake_soffice.mode("escape")
    monkeypatch.setattr(libreoffice, "_REAP_TIMEOUT_S", 0.5)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            libreoffice.convert(_src(tmp_path), "pptx", tmp_path, timeout=1)
        assert time.monotonic() - started < 10
    finally:
        os.kill(fake_soffice.grandchild_pid(), 9)


def test_group_already_gone_is_not_an_error(fake_soffice, tmp_path, monkeypatch):
    fake_soffice.mode("hang")
    real_killpg = os.killpg

    def killpg_then_lookup_error(pgid, sig):
        real_killpg(pgid, sig)
        raise ProcessLookupError

    monkeypatch.setattr(libreoffice.os, "killpg", killpg_then_lookup_error)
    with pytest.raises(RuntimeError, match="timed out"):
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path, timeout=1)


def test_interrupt_while_waiting_kills_the_group(fake_soffice, tmp_path, monkeypatch):
    """Any exception while waiting (e.g. KeyboardInterrupt) must not leave LibreOffice running."""
    fake_soffice.mode("hang")
    real_communicate = libreoffice.subprocess.Popen.communicate
    calls = {"n": 0}

    def interrupted(proc, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            deadline = time.monotonic() + 5
            while not fake_soffice.pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            raise KeyboardInterrupt
        return real_communicate(proc, *args, **kwargs)

    monkeypatch.setattr(libreoffice.subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        libreoffice.convert(_src(tmp_path), "pptx", tmp_path, timeout=30)
    grandchild = fake_soffice.grandchild_pid()
    deadline = time.monotonic() + 5
    while pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(grandchild)
