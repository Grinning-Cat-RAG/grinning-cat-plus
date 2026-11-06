"""Headless LibreOffice conversion, bounded in time and isolated per call.

Lifecycle: the caller owns ``workdir`` (and deletes it); this module writes into it the
conversion output (``workdir/out``) and a throw-away LibreOffice profile (``workdir/profile``).
The private profile lets concurrent conversions run side by side: with a shared profile a
second ``soffice`` hands its job to the first one and exits 0 without producing anything.
LibreOffice runs in its own process group, so on timeout (or any interruption while waiting)
the launcher *and* the ``soffice.bin`` it spawns are killed together.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

DEFAULT_TIMEOUT_S = 300.0


def libreoffice_binary() -> str | None:
    return shutil.which("libreoffice") or shutil.which("soffice")


def _kill(proc: subprocess.Popen) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:  # no process groups (Windows): only the direct child can be reached
        proc.kill()


def convert(source: Path, target_ext: str, workdir: Path, timeout: float = DEFAULT_TIMEOUT_S) -> Path:
    """Convert ``source`` to ``target_ext`` and return the converted file (inside ``workdir``).

    Raises:
        RuntimeError: LibreOffice is missing, fails, times out or produces no output.
    """
    binary = libreoffice_binary()
    if not binary:
        raise RuntimeError(
            "LibreOffice / soffice not found in PATH: install it (e.g. `apt-get install libreoffice`) "
            f"to ingest {source.suffix or 'these'} files."
        )

    outdir = workdir / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        f"-env:UserInstallation={(workdir / 'profile').as_uri()}",
        "--headless",
        "--convert-to",
        target_ext,
        "--outdir",
        str(outdir),
        str(source),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        proc.communicate()
        raise RuntimeError(f"LibreOffice conversion of {source.name} timed out after {timeout:g}s") from None
    except BaseException:
        _kill(proc)
        proc.communicate()
        raise

    if proc.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion of {source.name} failed (exit {proc.returncode}): "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
    converted = outdir / f"{source.stem}.{target_ext}"
    if not converted.exists():
        present = ", ".join(sorted(p.name for p in outdir.iterdir())) or "<none>"
        raise RuntimeError(f"Expected {converted.name} after LibreOffice conversion, found: {present}")
    return converted
