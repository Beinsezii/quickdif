import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

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


def test_quickdif_never_imports_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """All import checks in one function.
    Do not separate; pytest-xdist will likely race."""

    # check global imports
    assert quickdif.__file__ is not None
    assert not torches(), "Torch imported globally!"

    # check cli func. parse_cli() reads sys.argv, which is pytest's own argv unless we lie about it.
    monkeypatch.setattr(sys, "argv", ["quickdif"])
    quickdif.parse_cli(quickdif.Parameters(), False)
    quickdif.parse_cli(quickdif.Parameters(), True)
    assert not torches(), "Torch imported during CLI!"

    # check tester itself
    assert __import__("torch").__version__
    assert torches(), "Torch import was not caught!"
