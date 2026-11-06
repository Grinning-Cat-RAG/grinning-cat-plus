"""Shared fixtures.

The plugin directory is itself a Python package (``rabbithole.py`` uses relative imports), so the
tests import it under the name ``grinning_cat_plus``. The Grinning Cat core (``cat``) is not a
dependency of this repository: tests that need ``rabbithole`` install a minimal stub of it.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import stat
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "grinning_cat_plus"


def _load_package() -> None:
    if PACKAGE_NAME in sys.modules:
        return
    # The plugin has no __init__.py: an empty package rooted at the repository is enough.
    spec = importlib.machinery.ModuleSpec(PACKAGE_NAME, None, is_package=True)
    spec.submodule_search_locations = [str(REPO_ROOT)]
    sys.modules[PACKAGE_NAME] = importlib.util.module_from_spec(spec)


_load_package()


# --------------------------------------------------------------------------------------------
# Fake LibreOffice
# --------------------------------------------------------------------------------------------
# Modes (env FAKE_SOFFICE_MODE):
#   ok        -> writes <outdir>/<stem>.<ext> containing "CONVERTED:" + input bytes, exit 0
#   fail      -> prints to stderr, exit 3
#   nooutput  -> exit 0 without writing anything (the shared-profile symptom)
#   badbytes  -> exit 5 with non-UTF-8 bytes on stderr
#   hang      -> starts a grandchild that sleeps (pid in $FAKE_SOFFICE_PIDFILE), then becomes a sleep
#                itself (exec), like the soffice launcher waiting on soffice.bin
# Every invocation appends its argv to $FAKE_SOFFICE_ARGS (one line per call).
_FAKE_SOFFICE = textwrap.dedent(
    """\
    #!/bin/bash
    # The test PATH holds only this script (so no real LibreOffice is ever picked up).
    export PATH=/usr/bin:/bin
    printf '%s\\n' "$*" >> "${FAKE_SOFFICE_ARGS:-/dev/null}"
    outdir=""; ext=""; src=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --outdir) outdir="$2"; shift 2 ;;
        --convert-to) ext="$2"; shift 2 ;;
        *) src="$1"; shift ;;
      esac
    done
    case "${FAKE_SOFFICE_MODE:-ok}" in
      ok)
        stem="$(basename "$src")"; stem="${stem%.*}"
        { printf 'CONVERTED:'; cat "$src"; } > "$outdir/$stem.$ext"
        exit 0 ;;
      fail) echo "boom" >&2; exit 3 ;;
      nooutput) exit 0 ;;
      badbytes) printf '\\xff\\xfe' >&2; exit 5 ;;
      hang)
        sleep 600 &
        echo $! > "$FAKE_SOFFICE_PIDFILE"
        exec sleep 600 ;;
    esac
    """
)


@pytest.fixture
def fake_soffice(tmp_path, monkeypatch):
    """Put a fake ``soffice`` (and no ``libreoffice``) first on PATH."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    script = bindir / "soffice"
    script.write_text(_FAKE_SOFFICE)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("FAKE_SOFFICE_ARGS", str(tmp_path / "soffice-args.txt"))
    monkeypatch.setenv("FAKE_SOFFICE_PIDFILE", str(tmp_path / "grandchild.pid"))
    monkeypatch.setenv("FAKE_SOFFICE_MODE", "ok")

    class Fake:
        args_file = tmp_path / "soffice-args.txt"
        pid_file = tmp_path / "grandchild.pid"

        @staticmethod
        def mode(value: str) -> None:
            monkeypatch.setenv("FAKE_SOFFICE_MODE", value)

        @staticmethod
        def calls() -> list[str]:
            if not Fake.args_file.exists():
                return []
            return Fake.args_file.read_text().splitlines()

    return Fake


@pytest.fixture
def isolated_tmp(tmp_path, monkeypatch):
    """Redirect ``tempfile`` to an empty directory so leaked temp entries can be counted."""
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0): treat it as dead.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return False
