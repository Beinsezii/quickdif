import sys
from importlib import import_module
from typing import Any

IMPORTED: set[str] = set()


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if event == "import":
        IMPORTED.add(args[0])


sys.addaudithook(_audit)

lazy import quickdif  # noqa: E402


def torches() -> list[str]:
    # ledger catches import-then-del, sys.modules catches anything loaded before the hook landed.
    return sorted(m for m in IMPORTED | set(sys.modules) if m == "torch" or m.startswith("torch."))


def test_audit_hook_records() -> None:
    # audit fires even on failed lookups, so this proves the hook works without importing anything real
    try:
        import_module("not_a_real_module_zzz")
    except ModuleNotFoundError:
        pass
    assert "not_a_real_module_zzz" in IMPORTED


def test_quickdif_never_imports_torch() -> None:
    assert quickdif.__file__ is not None
    assert not torches()
    # prove the detector isn't vacuously passing by importing torch for real and catching it.
    # must stay in this same test function or pytest-xdist would race on import
    assert __import__("torch").__version__
    assert torches()
