import re


def train_text_normalization(s: str) -> str:
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    s = s.replace("‘", "'")
    s = s.replace("’", "'")
    if s[:2] == "\" ": # remove the starting double quote
        s = s[2:]

    return s


def ref_text_normalization(ref_text: str) -> str:
    # Rule 1: Remove the [FN#[]]
    p = r"[FN#[0-9]*]"
    pattern = re.compile(p)

    # ref_text = ref_text.replace("”", "\"")
    # ref_text = ref_text.replace("’", "'")
    res = pattern.findall(ref_text)
    ref_text = re.sub(p, "", ref_text)
    
    ref_text = train_text_normalization(ref_text)

    return ref_text


def remove_non_alphabetic(text: str, strict: bool=True) -> str:
    if not strict:
        # Note, this also keeps space, single quote(') and hypen (-)
        text = text.replace("-", " ")
        text = text.replace("—", " ")
        return re.sub("[^a-zA-Z0-9\s']+", "", text)
    else:
        # only keeps space
        return re.sub("[^a-zA-Z\s]+", "", text)


def recog_text_normalization(recog_text: str) -> str:
    pass

def upper_only_alpha(text: str) -> str:
    return remove_non_alphabetic(text.upper(), strict=False)

def lower_only_alpha(text: str) -> str:
    return remove_non_alphabetic(text.lower(), strict=False)

def lower_all_char(text: str) -> str:
    return text.lower()

def upper_all_char(text: str) -> str:
    return text.upper()

if __name__ == "__main__":
    ref_text = "Mixed-case English transcription, with punctuation. Actually, it is fully not related."
    print(ref_text)
    res = upper_only_alpha(ref_text)
    print(res)
