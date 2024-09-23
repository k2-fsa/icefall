from num2words import num2words


def remove_punc_to_upper(text: str) -> str:
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'")
    s_list = [x.upper() if x in tokens else " " for x in text]
    s = " ".join("".join(s_list).split()).strip()
    return s


def word_normalization(word: str) -> str:
    # 1. Use full word for some abbreviation
    # 2. Convert digits to english words
    # 3. Convert ordinal number to english words
    if word == "MRS":
        return "MISSUS"
    if word == "MR":
        return "MISTER"
    if word == "ST":
        return "SAINT"
    if word == "ECT":
        return "ET CETERA"

    if word[-2:] in ("ST", "ND", "RD", "TH") and word[:-2].isnumeric():  #  e.g 9TH, 6TH
        word = num2words(word[:-2], to="ordinal")
        word = word.replace("-", " ")

    if word.isnumeric():
        num = int(word)
        if num > 1500 and num < 2030:
            word = num2words(word, to="year")
        else:
            word = num2words(word)
        word = word.replace("-", " ")
    return word.upper()


def text_normalization(text: str) -> str:
    text = text.upper()
    return " ".join([word_normalization(x) for x in text.split()])


if __name__ == "__main__":
    assert remove_punc_to_upper("I like this 《book>") == "I LIKE THIS BOOK"
    assert (
        text_normalization("Hello Mrs st 21st world 3rd she 99th MR")
        == "HELLO MISSUS SAINT TWENTY FIRST WORLD THIRD SHE NINETY NINTH MISTER"
    )
