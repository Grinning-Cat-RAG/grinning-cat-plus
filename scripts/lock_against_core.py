"""Regenerate this plugin's uv.lock against the versions pinned by the Grinning Cat core.

The core installs a plugin that ships a uv.lock strictly from it, without ever regenerating it, and refuses it when it
is out of date with pyproject.toml or would install a version of a library other than the one pinned in the core's own
uv.lock (all the plugins share the core's virtual environment). A plain `uv lock` picks the newest versions, which
usually differ from the core's: this script resolves the lock with every package of the core's uv.lock as a
constraint, as the core itself does when it compiles a plugin that ships no lock.

The existing uv.lock is used as the starting point, so versions change only where needed.

Usage, from the plugin's root folder:

    python scripts/lock_against_core.py /path/to/grinning-cat-core/uv.lock

Run it again whenever pyproject.toml or the core's uv.lock changes. Use `uv sync --frozen` locally afterwards: a plain
`uv sync` would re-lock without the core's constraints.

The core imports every .py file of a plugin, so nothing runs unless the script is executed directly.
"""

if __name__ == "__main__":
    import shutil
    import subprocess
    import sys
    import tempfile
    import tomllib
    from pathlib import Path

    def main() -> int:
        if len(sys.argv) != 2:
            print(__doc__)
            return 2
        core_lock = Path(sys.argv[1])
        plugin_root = Path(__file__).resolve().parent.parent
        pyproject = plugin_root / "pyproject.toml"
        plugin_lock = plugin_root / "uv.lock"

        pins = [
            f"{package['name']}=={package['version']}"
            for package in tomllib.loads(core_lock.read_text())["package"]
            if "version" in package
        ]
        pyproject_text = pyproject.read_text()
        if "uv" in tomllib.loads(pyproject_text).get("tool", {}):
            print("pyproject.toml has a [tool.uv] table: merge the core's pins into it by hand", file=sys.stderr)
            return 1

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            constraints = "".join(f'    "{pin}",\n' for pin in pins)
            (tmp_dir / "pyproject.toml").write_text(
                pyproject_text + f"\n[tool.uv]\nconstraint-dependencies = [\n{constraints}]\n"
            )
            if plugin_lock.exists():
                shutil.copyfile(plugin_lock, tmp_dir / "uv.lock")
            result = subprocess.run(["uv", "lock", "--project", str(tmp_dir)], check=False)
            if result.returncode != 0:
                print("uv lock failed: this plugin cannot be resolved next to the core's pins", file=sys.stderr)
                return result.returncode
            shutil.copyfile(tmp_dir / "uv.lock", plugin_lock)

        print(f"uv.lock resolved against {len(pins)} pins of {core_lock}")
        return 0

    sys.exit(main())
