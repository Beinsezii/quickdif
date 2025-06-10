from collections.abc import Callable
from typing import Any

from quickdif import Grid, Resolution, loras_to_str, pexpand, splitlist


def fails(f: Callable[[], Any]) -> bool:
    try:
        f()
    except Exception:  # noqa: BLE001
        return True
    return False


def assertsort(l1: list, l2: list) -> None:
    l1.sort()
    l2.sort()
    assert l1 == l2


def test_lora_to_str() -> None:
    # simple floats should be str() stable
    assert loras_to_str([]) is None
    assert loras_to_str(["lora_1:::0.5"]) == "lora_1:::0.5"
    assert loras_to_str(["lora_2:::0.0"]) is None
    assert loras_to_str(["lora_3:::1.0"]) == "lora_3"
    assert loras_to_str(["lora_4"]) == "lora_4"
    assert loras_to_str(["lora_5:::-0"]) is None
    assert loras_to_str(["lora_6:::-1"]) == "lora_6:::-1.0"
    assert (
        loras_to_str(["lora_1:::0.5", "lora_2:::0.0", "lora_3:::1.0", "lora_4"]) == "lora_1:::0.5\x1flora_3\x1flora_4"
    )


def test_splitlist() -> None:
    assert splitlist([]) == []
    assert splitlist(["a"]) == [["a"]]
    assert splitlist(["a:::0.1"]) == [["a:::0.1"]]
    assert splitlist(["a", "b"]) == [["a", "b"]]
    assert splitlist(["a", ":::", "b"]) == [["a"], ["b"]]
    assert splitlist([":::", "a", "b"]) == [[], ["a", "b"]]
    assert splitlist(["a", "b", ":::"]) == [["a", "b"], []]
    assert splitlist([":::", ":::"]) == [[], [], []]
    assert splitlist(["a", ":::", "", "b", ":::", "c", "d", ":::"]) == [["a"], ["b"], ["c", "d"], []]
    assert splitlist(["a", ":::", "", "b", ":::", "c", "d", ":::"], trim_groups=True) == [["a"], ["b"], ["c", "d"]]
    assert splitlist(["a", ":::", "", "b", ":::", "c", "d", ":::"], trim_items=False) == [
        ["a"],
        ["", "b"],
        ["c", "d"],
        [],
    ]


def test_resolution_basic() -> None:
    assert Resolution("1920x1080").resolution == (1920, 1080)
    assert Resolution("1920 * 1080").resolution == (1920, 1080)
    assert Resolution("1920 1080").resolution == (1920, 1080)
    assert Resolution("x1536").resolution == (1024, 1536)
    assert Resolution("1536").resolution == (1536, 1024)
    assert Resolution("1536 x").resolution == (1536, 1024)


def test_resolution_aspect() -> None:
    assert Resolution("1:1@1.0").resolution == (1024, 1024)
    assert Resolution("4 : 1 ^ 1024").resolution == (2048, 512)
    assert Resolution("1:1:1 @4").resolution == (2000, 2000)


def test_resolution_invalid() -> None:
    assert not fails(lambda: Resolution("1920x1080"))
    assert fails(lambda: Resolution(""))
    assert fails(lambda: Resolution("x"))
    assert fails(lambda: Resolution(":"))
    assert fails(lambda: Resolution("bunny"))
    assert fails(lambda: Resolution("5x5x5"))
    assert fails(lambda: Resolution("1@5.0"))
    assert fails(lambda: Resolution("1::1@1"))
    assert fails(lambda: Resolution("1:1:1:1"))
    assert fails(lambda: Resolution("1:1:1:1@1"))


def test_grid_valid() -> None:
    assert Grid("Prompt Seed").axes == ("prompt", "seed")
    assert Grid("none, guidance").axes == (None, "guidance")
    assert Grid("posterize:pixelate").axes == ("posterize", "pixelate")
    assert Grid(("steps", None)).axes == ("steps", None)
    assert Grid("variance_scale").axes == ("variance_scale", None)
    assert Grid(":color_power").axes == (None, "color_power")


def test_grid_invalid() -> None:
    assert not fails(lambda: Grid("Prompt Seed"))
    assert fails(lambda: Grid(""))
    assert fails(lambda: Grid(","))
    assert fails(lambda: Grid("Prompt Seed Steps"))
    assert fails(lambda: Grid("prompt succulent"))
    assert fails(lambda: Grid("flowering"))
    assert fails(lambda: Grid("seed, grid"))
    assert fails(lambda: Grid("batch_count"))
    assert fails(lambda: Grid("prompt x seed"))


def test_pexpand_basic() -> None:
    assertsort(pexpand("{a|b}"), ["a", "b"])


def test_pexpand_multi() -> None:
    assertsort(pexpand("{ ab |c|d e}"), [" ab ", "c", "d e"])


def test_pexpand_empty() -> None:
    assertsort(pexpand("foo {a ||b }bar"), ["foo a bar", "foo bar", "foo b bar"])


def test_pexpand_escape() -> None:
    assertsort(pexpand("foo \\{a||b}"), ["foo {a||b}"])


def test_pexpand_escape2() -> None:
    assertsort(pexpand("foo \\\\{a||b}"), ["foo \\a", "foo \\", "foo \\b"])


def test_pexpand_recurse() -> None:
    assertsort(
        pexpand("the{{{}\\}} {wet|{big|smol}{ soft|}} {{{}}}dog"),
        ["the{} wet dog", "the{} big soft dog", "the{} smol soft dog", "the{} big dog", "the{} smol dog"],
    )
