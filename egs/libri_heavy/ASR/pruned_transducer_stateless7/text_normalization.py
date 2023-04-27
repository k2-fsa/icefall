import re


def ref_text_normalization(ref_text: str) -> str:
    # Rule 1: Remove the [FN#[]]
    p = r"[FN#[0-9]*]"
    pattern = re.compile(p)

    # ref_text = ref_text.replace("”", "\"")
    # ref_text = ref_text.replace("’", "'")
    res = pattern.findall(ref_text)
    ref_text = re.sub(p, "", ref_text)

    return ref_text.lower()


def remove_non_alphabetic(text: str) -> str:
    return re.sub("[^a-zA-Z]+", "", text)


def recog_text_normalization(recog_text: str) -> str:
    pass


if __name__ == "__main__":
    ref_text = " Quoth Shahrazad [FN#1]:—It hath reached me, O [FN#987]auspicious King of intelligence "
    res = ref_text_normalization(ref_text)
    print(res)
