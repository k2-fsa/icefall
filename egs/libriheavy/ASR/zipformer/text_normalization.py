import re

def replace_full_width_symbol(s: str) -> str:
    # replace full-width symbol with theri half width counterpart
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    s = s.replace("‘", "'")
    s = s.replace("’", "'")

    return s


def upper_ref_text(text: str) -> str:
    text = replace_full_width_symbol(text)
    text = text.upper() # upper case all characters
    
    # Only keep all alpha-numeric characters, hypen and apostrophe
    text = text.replace("--", " ")
    text = re.sub("[^a-zA-Z0-9\s\'-]+", "", text)
    return text

def simple_normalization(text: str) -> str:
    text = replace_full_width_symbol(text)
    text = text.replace("--", " ")
    
    return text
    