# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
import re
import string


def remove_punctuation(s: str) -> str:
    """
    It implements https://github.com/espnet/espnet/blob/master/utils/remove_punctuation.pl
    """

    # Remove punctuation except apostrophe
    # s/<space>/spacemark/g;  # for scoring
    s = re.sub("<space>", "spacemark", s)

    # s/'/apostrophe/g;
    s = re.sub("'", "apostrophe", s)

    # s/[[:punct:]]//g;
    s = s.translate(str.maketrans("", "", string.punctuation))
    # string punctuation returns the following string
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # See
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string

    # s/apostrophe/'/g;
    s = re.sub("apostrophe", "'", s)

    # s/spacemark/<space>/g;  # for scoring
    s = re.sub("spacemark", "<space>", s)

    # remove whitespace
    # s/\s+/ /g;
    s = re.sub("\s+", " ", s)

    # s/^\s+//;
    s = re.sub("^\s+", "", s)

    # s/\s+$//;
    s = re.sub("\s+$", "", s)

    return s
