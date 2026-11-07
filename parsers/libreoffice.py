"""Headless LibreOffice conversion, bounded in time and isolated per call.

Lifecycle: the caller owns ``workdir`` (and deletes it); this module writes into it the
conversion output (``workdir/out``) and a throw-away LibreOffice profile (``workdir/profile``).
The private profile lets concurrent conversions run side by side: with a shared profile a
second ``soffice`` hands its job to the first one and exits 0 without producing anything.
LibreOffice runs in its own process group, so on timeout (or any interruption while waiting)
the launcher *and* the ``soffice.bin`` it spawns are killed together. POSIX only (process
groups): the plugin runs in a Linux container.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

DEFAULT_TIMEOUT_S = 300.0
# After a kill, how long to wait for the pipes to drain: a descendant outside the killed
# process group (or any descendant where process groups do not exist) may keep them open.
_REAP_TIMEOUT_S = 5.0


def libreoffice_binary() -> str | None:
    return shutil.which("libreoffice") or shutil.which("soffice")


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # A surviving descendant still holds the pipes: stop reading, the child itself is dead.
        for pipe in (proc.stdout, proc.stderr):
            pipe.close()
        proc.wait()


def convert(
    source: Path,
    target_ext: str,
    workdir: Path,
    timeout: float = DEFAULT_TIMEOUT_S,
    display_name: str | None = None,
) -> Path:
    """Convert ``source`` to ``target_ext`` and return the converted file (inside ``workdir``).

    ``display_name`` is how errors name the file (the user's file, not the internal copy).

    Raises:
        RuntimeError: LibreOffice is missing, fails, times out or produces no output.
    """
    name = display_name or source.name
    binary = libreoffice_binary()
    if not binary:
        raise RuntimeError(
            "LibreOffice / soffice not found in PATH: install it (e.g. `apt-get install libreoffice`) "
            f"to ingest {name}."
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
        raise RuntimeError(f"LibreOffice conversion of {name} timed out after {timeout:g}s") from None
    except BaseException:
        _kill(proc)
        raise

    if proc.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion of {name} failed (exit {proc.returncode}): "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
    converted = outdir / f"{source.stem}.{target_ext}"
    if not converted.exists():
        # soffice exits 0 also when it cannot load the input: its output is the only diagnostic.
        raise RuntimeError(
            f"LibreOffice conversion of {name} produced no {target_ext} file: "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
    return converted
