import re

words = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
ordinal_nums = [
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
]

num_ordinal_dict = {num: ordinal_nums[num] for num in range(21)}


def year_to_words(num: int):
    assert isinstance(num, int), num
    # check if a num is representing a year
    if num > 1500 and num < 2000:
        return words[num // 100] + " " + num_to_words(num % 100)
    elif num == 2000:
        return "TWO THOUSAND"
    elif num > 2000:
        return "TWO THOUSAND AND " + num_to_words(num % 100)
    else:
        return num_to_words(num)


def num_to_words(num: int):
    # Return the English words of a integer number

    # If this is a year number
    if num > 1500 and num < 2030:
        return year_to_words(num)

    if num < 20:
        return words[num]
    if num < 100:
        if num % 10 == 0:
            return words[num // 10 * 10]
        else:
            return words[num // 10 * 10] + " " + words[num % 10]
    if num < 1000:
        return words[num // 100] + " hundred and " + num_to_words(num % 100)
    if num < 1000000:
        return num_to_words(num // 1000) + " thousand " + num_to_words(num % 1000)
    return num


def num_to_ordinal_word(num: int):

    return num_ordinal_dict.get(num, num_to_words(num)).upper()


def replace_full_width_symbol(s: str) -> str:
    # replace full-width symbol with theri half width counterpart
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    s = s.replace("‘", "'")
    s = s.replace("’", "'")

    return s


def decoding_normalization(text: str) -> str:
    text = replace_full_width_symbol(text)

    # Only keep all alpha-numeric characters, hypen and apostrophe
    text = text.replace("-", " ")
    text = re.sub(r"[^a-zA-Z0-9\s']+", "", text)
    return text


def word_normalization(word: str) -> str:
    # 1 .Use full word for some abbreviation
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
    if word.isnumeric():
        word = num_to_words(int(word))
        return str(word).upper()
    #  e.g 9TH, 6TH
    if word[-2:] == "TH" and word[0].isnumeric():
        return num_to_ordinal_word(int(word[:-2])).upper()
    if word[0] == "'":
        return word[1:]

    return word


def simple_normalization(text: str) -> str:
    text = replace_full_width_symbol(text)
    text = text.replace("--", " ")

    return text


if __name__ == "__main__":

    s = str(1830)
    out = word_normalization(s)
    print(s, out)
