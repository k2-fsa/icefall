# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
import re


def normalize_punctuation(s: str, lang: str) -> str:
    """
    This function implements
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/normalize-punctuation.perl

    Args:
      s:
        A string to be normalized.
      lang:
        The language to which `s` belongs
    Returns:
      Return a normalized string.
    """
    #  s/\r//g;
    s = re.sub("\r", "", s)

    # remove extra spaces
    # s/\(/ \(/g;
    s = re.sub("\(", " (", s)  # add a space before (

    # s/\)/\) /g; s/ +/ /g;
    s = re.sub("\)", ") ", s)  # add a space after )
    s = re.sub(" +", " ", s)  # convert multiple spaces to one

    # s/\) ([\.\!\:\?\;\,])/\)$1/g;
    s = re.sub("\) ([\.\!\:\?\;\,])", r")\1", s)

    # s/\( /\(/g;
    s = re.sub("\( ", "(", s)  # remove space after (

    # s/ \)/\)/g;
    s = re.sub(" \)", ")", s)  # remove space before )

    # s/(\d) \%/$1\%/g;
    s = re.sub("(\d) \%", r"\1%", s)  # remove space between a digit and %

    # s/ :/:/g;
    s = re.sub(" :", ":", s)  # remove space before :

    # s/ ;/;/g;
    s = re.sub(" ;", ";", s)  # remove space before ;

    # normalize unicode punctuation
    # s/\`/\'/g;
    s = re.sub("`", "'", s)  # replace ` with '

    # s/\'\'/ \" /g;
    s = re.sub("''", '"', s)  #  replace '' with "

    # s/„/\"/g;
    s = re.sub("„", '"', s)  #  replace „ with "

    # s/“/\"/g;
    s = re.sub("“", '"', s)  #  replace “ with "

    # s/”/\"/g;
    s = re.sub("”", '"', s)  #  replace ” with "

    # s/–/-/g;
    s = re.sub("–", "-", s)  #  replace – with -

    # s/—/ - /g; s/ +/ /g;
    s = re.sub("—", " - ", s)
    s = re.sub(" +", " ", s)  # convert multiple spaces to one

    # s/´/\'/g;
    s = re.sub("´", "'", s)

    # s/([a-z])‘([a-z])/$1\'$2/gi;
    s = re.sub("([a-z])‘([a-z])", r"\1'\2", s, flags=re.IGNORECASE)

    # s/([a-z])’([a-z])/$1\'$2/gi;
    s = re.sub("([a-z])’([a-z])", r"\1'\2", s, flags=re.IGNORECASE)

    # s/‘/\'/g;
    s = re.sub("‘", "'", s)

    # s/‚/\'/g;
    s = re.sub("‚", "'", s)

    # s/’/\"/g;
    s = re.sub("’", '"', s)

    # s/''/\"/g;
    s = re.sub("''", '"', s)

    # s/´´/\"/g;
    s = re.sub("´´", '"', s)

    # s/…/.../g;
    s = re.sub("…", "...", s)

    # French quotes

    # s/ « / \"/g;
    s = re.sub(" « ", ' "', s)

    # s/« /\"/g;
    s = re.sub("« ", '"', s)

    # s/«/\"/g;
    s = re.sub("«", '"', s)

    # s/ » /\" /g;
    s = re.sub(" » ", '" ', s)

    # s/ »/\"/g;
    s = re.sub(" »", '"', s)

    # s/»/\"/g;
    s = re.sub("»", '"', s)

    # handle pseudo-spaces

    # s/ \%/\%/g;
    s = re.sub(" %", r"%", s)

    # s/nº /nº /g;
    s = re.sub("nº ", "nº ", s)

    # s/ :/:/g;
    s = re.sub(" :", ":", s)

    # s/ ºC/ ºC/g;
    s = re.sub(" ºC", " ºC", s)

    # s/ cm/ cm/g;
    s = re.sub(" cm", " cm", s)

    # s/ \?/\?/g;
    s = re.sub(" \?", "\?", s)

    # s/ \!/\!/g;
    s = re.sub(" \!", "\!", s)

    # s/ ;/;/g;
    s = re.sub(" ;", ";", s)

    # s/, /, /g; s/ +/ /g;
    s = re.sub(", ", ", ", s)
    s = re.sub(" +", " ", s)

    if lang == "en":
        # English "quotation," followed by comma, style
        # s/\"([,\.]+)/$1\"/g;
        s = re.sub('"([,\.]+)', r'\1"', s)
    elif lang in ("cs", "cz"):
        # Czech is confused
        pass
    else:
        # German/Spanish/French "quotation", followed by comma, style
        # s/,\"/\",/g;
        s = re.sub(',"', '",', s)

        # s/(\.+)\"(\s*[^<])/\"$1$2/g; # don't fix period at end of sentence
        s = re.sub('(\.+)"(\s*[^<])', r'"\1\2', s)

    if lang in ("de", "es", "cz", "cs", "fr"):
        # s/(\d) (\d)/$1,$2/g;
        s = re.sub("(\d) (\d)", r"\1,\2", s)
    else:
        # s/(\d) (\d)/$1.$2/g;
        s = re.sub("(\d) (\d)", r"\1.\2", s)

    return s
