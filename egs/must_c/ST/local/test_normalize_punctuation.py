#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

from normalize_punctuation import normalize_punctuation


def test_normalize_punctuation():
    #  s/\r//g;
    s = "a\r\nb\r\n"
    n = normalize_punctuation(s, lang="en")
    assert "\r" not in n
    assert len(s) - 2 == len(n), (len(s), len(n))

    # s/\(/ \(/g;
    s = "(ab (c"
    n = normalize_punctuation(s, lang="en")
    assert n == " (ab (c", n

    # s/\)/\) /g;
    s = "a)b c)"
    n = normalize_punctuation(s, lang="en")
    assert n == "a) b c) "

    # s/ +/ /g;
    s = "  a  b     c  d    "
    n = normalize_punctuation(s, lang="en")
    assert n == " a b c d "

    # s/\) ([\.\!\:\?\;\,])/\)$1/g;
    for i in ".!:?;,":
        s = f"a)  {i}"
        n = normalize_punctuation(s, lang="en")
        assert n == f"a){i}"

    # s/\( /\(/g;
    s = "a(    b"
    n = normalize_punctuation(s, lang="en")
    assert n == "a (b", n

    # s/ \)/\)/g;
    s = "ab    ) a"
    n = normalize_punctuation(s, lang="en")
    assert n == "ab) a", n

    # s/(\d) \%/$1\%/g;
    s = "1   %a"
    n = normalize_punctuation(s, lang="en")
    assert n == "1%a", n

    # s/ :/:/g;
    s = "a  :"
    n = normalize_punctuation(s, lang="en")
    assert n == "a:", n

    # s/ ;/;/g;
    s = "a  ;"
    n = normalize_punctuation(s, lang="en")
    assert n == "a;", n

    # s/\`/\'/g;
    s = "`a`"
    n = normalize_punctuation(s, lang="en")
    assert n == "'a'", n

    # s/\'\'/ \" /g;
    s = "''a''"
    n = normalize_punctuation(s, lang="en")
    assert n == '"a"', n

    # s/„/\"/g;
    s = '„a"'
    n = normalize_punctuation(s, lang="en")
    assert n == '"a"', n

    # s/“/\"/g;
    s = "“a„"
    n = normalize_punctuation(s, lang="en")
    assert n == '"a"', n

    # s/”/\"/g;
    s = "“a”"
    n = normalize_punctuation(s, lang="en")
    assert n == '"a"', n

    # s/–/-/g;
    s = "a–b"
    n = normalize_punctuation(s, lang="en")
    assert n == "a-b", n

    # s/—/ - /g; s/ +/ /g;
    s = "a—b"
    n = normalize_punctuation(s, lang="en")
    assert n == "a - b", n

    # s/´/\'/g;
    s = "a´b"
    n = normalize_punctuation(s, lang="en")
    assert n == "a'b", n

    # s/([a-z])‘([a-z])/$1\'$2/gi;
    for i in "‘’":
        s = f"a{i}B"
        n = normalize_punctuation(s, lang="en")
        assert n == "a'B", n

        s = f"A{i}B"
        n = normalize_punctuation(s, lang="en")
        assert n == "A'B", n

        s = f"A{i}b"
        n = normalize_punctuation(s, lang="en")
        assert n == "A'b", n

    # s/‘/\'/g;
    # s/‚/\'/g;
    for i in "‘‚":
        s = f"a{i}b"
        n = normalize_punctuation(s, lang="en")
        assert n == "a'b", n

    # s/’/\"/g;
    s = "’"
    n = normalize_punctuation(s, lang="en")
    assert n == '"', n

    # s/''/\"/g;
    s = "''"
    n = normalize_punctuation(s, lang="en")
    assert n == '"', n

    # s/´´/\"/g;
    s = "´´"
    n = normalize_punctuation(s, lang="en")
    assert n == '"', n

    # s/…/.../g;
    s = "…"
    n = normalize_punctuation(s, lang="en")
    assert n == "...", n

    # s/ « / \"/g;
    s = "a « b"
    n = normalize_punctuation(s, lang="en")
    assert n == 'a "b', n

    # s/« /\"/g;
    s = "a « b"
    n = normalize_punctuation(s, lang="en")
    assert n == 'a "b', n

    # s/«/\"/g;
    s = "a«b"
    n = normalize_punctuation(s, lang="en")
    assert n == 'a"b', n

    # s/ » /\" /g;
    s = " » "
    n = normalize_punctuation(s, lang="en")
    assert n == '" ', n

    # s/ »/\"/g;
    s = " »"
    n = normalize_punctuation(s, lang="en")
    assert n == '"', n

    # s/»/\"/g;
    s = "»"
    n = normalize_punctuation(s, lang="en")
    assert n == '"', n

    # s/ \%/\%/g;
    s = " %"
    n = normalize_punctuation(s, lang="en")
    assert n == "%", n

    # s/ :/:/g;
    s = " :"
    n = normalize_punctuation(s, lang="en")
    assert n == ":", n

    # s/(\d) (\d)/$1.$2/g;
    s = "2 3"
    n = normalize_punctuation(s, lang="en")
    assert n == "2.3", n

    # s/(\d) (\d)/$1,$2/g;
    s = "2 3"
    n = normalize_punctuation(s, lang="de")
    assert n == "2,3", n


def main():
    test_normalize_punctuation()


if __name__ == "__main__":
    main()
