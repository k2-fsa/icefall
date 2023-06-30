#!/usr/bin/env python3

from remove_punctuation import remove_punctuation


def test_remove_punctuation():
    s = "a,b'c!#"
    n = remove_punctuation(s)
    assert n == "ab'c", n

    s = "  ab  "  # remove leading and trailing spaces
    n = remove_punctuation(s)
    assert n == "ab", n


if __name__ == "__main__":
    test_remove_punctuation()
