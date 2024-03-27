from quickdif import pexpand


def assertsort(l1, l2):
    l1.sort()
    l2.sort()
    assert l1 == l2


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
