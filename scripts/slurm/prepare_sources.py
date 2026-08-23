"""Materialize the patched upstream tree locally, without Modal.

The SafeSelfPlay patch helpers live inside Modal launcher modules, but the
helpers themselves are pure file rewrites. This stubs `modal` so the modules
import, repoints their UPSTREAM_SOURCE/UPSTREAM_WORK constants at a local
checkout, and runs the same aggregate patch entry point the Modal image runs.

    git clone https://github.com/mickelliu/selfplay-redteaming "$SSP_ROOT/upstream"
    git -C "$SSP_ROOT/upstream" checkout 0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123
    SSP_ROOT=/path/to/workspace python scripts/slurm/prepare_sources.py

Writes the patched tree to $SSP_WORK (default $SSP_ROOT/work).
"""

from __future__ import annotations

import os
import pathlib
import sys
import types


def _stub_modal() -> None:
    """Absorb Image.from_registry(...).pip_install(...) and @app.function(...)."""

    class _Any:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, _name):
            return _Any()

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]          # behave as a pass-through decorator
            return _Any()

        def __or__(self, _other):
            return _Any()

    stub = types.ModuleType("modal")
    stub.__getattr__ = lambda _name: _Any()
    for attr in ("App", "Image", "Secret", "Volume", "Function", "Sandbox",
                 "Cls", "Dict", "Queue"):
        setattr(stub, attr, _Any())
    sys.modules.setdefault("modal", stub)
    for extra in ("modal.runner", "modal.io_streams"):
        sys.modules.setdefault(extra, types.ModuleType(extra))


def main() -> int:
    root = os.environ.get("SSP_ROOT")
    if not root:
        print("set SSP_ROOT to the workspace directory", file=sys.stderr)
        return 2
    source = pathlib.Path(os.environ.get("SSP_UPSTREAM_SRC", f"{root}/upstream"))
    work = pathlib.Path(os.environ.get("SSP_WORK", f"{root}/work"))
    if not source.is_dir():
        print(f"missing upstream checkout: {source}", file=sys.stderr)
        return 2

    _stub_modal()
    repo = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))

    import modal_upstream_selfredteam_role_lora_v2 as v2  # noqa: F401
    import modal_upstream_selfredteam_fixed_seed  # noqa: F401
    import modal_upstream_selfredteam_role_lora  # noqa: F401
    import modal_upstream_selfredteam_role_full  # noqa: F401

    # Each launcher module carries its own copy of these constants; rebind all of
    # them so the patch chain reads and writes the local checkout rather than the
    # image paths (/selfplay-redteaming, /tmp/selfplay-redteaming).
    for name, module in list(sys.modules.items()):
        if not name.startswith("modal_"):
            continue
        if hasattr(module, "UPSTREAM_SOURCE"):
            module.UPSTREAM_SOURCE = source
        if hasattr(module, "UPSTREAM_WORK"):
            module.UPSTREAM_WORK = work

    v2._prepare_lora_v2_upstream()
    print(f"prepared: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
