from quickdif import Grid, Resolution, pexpand


def fails(f) -> bool:
    try:
        f()
    except Exception:
        return True
    return False


def assertsort(l1, l2):
    l1.sort()
    l2.sort()
    assert l1 == l2


# pexpand {{{
def test_pexpand_basic():
    assertsort(pexpand("{a|b}"), ["a", "b"])


def test_pexpand_multi():
    assertsort(pexpand("{ ab |c|d e}"), [" ab ", "c", "d e"])


def test_pexpand_empty():
    assertsort(pexpand("foo {a ||b }bar"), ["foo a bar", "foo bar", "foo b bar"])


def test_pexpand_escape():
    assertsort(pexpand("foo \\{a||b}"), ["foo {a||b}"])


def test_pexpand_escape2():
    assertsort(pexpand("foo \\\\{a||b}"), ["foo \\a", "foo \\", "foo \\b"])


def test_pexpand_recurse():
    assertsort(
        pexpand("the{{{}\\}} {wet|{big|smol}{ soft|}} {{{}}}dog"),
        ["the{} wet dog", "the{} big soft dog", "the{} smol soft dog", "the{} big dog", "the{} smol dog"],
    )


# pexpand }}}


# Resolution {{{
def test_resolution_basic():
    assert Resolution("1920x1080").resolution == (1920, 1080)
    assert Resolution("1920 * 1080").resolution == (1920, 1080)
    assert Resolution("1920 1080").resolution == (1920, 1080)
    assert Resolution("x1536").resolution == (1024, 1536)
    assert Resolution("1536").resolution == (1536, 1024)
    assert Resolution("1536 x").resolution == (1536, 1024)


def test_resolution_aspect():
    assert Resolution("1:1@1.0").resolution == (1024, 1024)
    assert Resolution("4 : 1 ^ 1024").resolution == (2048, 512)
    assert Resolution("1:1:1 @4").resolution == (2000, 2000)


def test_resolution_invalid():
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


# }}}


# Grid {{{
def test_grid_valid():
    assert Grid("Prompt Seed").axes == ("prompt", "seed")
    assert Grid("none, guidance").axes == (None, "guidance")
    assert Grid("posterize:pixelate").axes == ("posterize", "pixelate")
    assert Grid(("steps", None)).axes == ("steps", None)
    assert Grid("variance_scale").axes == ("variance_scale", None)
    assert Grid(":color_power").axes == (None, "color_power")


def test_grid_invalid():
    assert not fails(lambda: Grid("Prompt Seed"))
    assert fails(lambda: Grid(""))
    assert fails(lambda: Grid(","))
    assert fails(lambda: Grid("Prompt Seed Steps"))
    assert fails(lambda: Grid("prompt succulent"))
    assert fails(lambda: Grid("flowering"))
    assert fails(lambda: Grid("seed, grid"))
    assert fails(lambda: Grid("batch_count"))
    assert fails(lambda: Grid("prompt x seed"))


# }}}
