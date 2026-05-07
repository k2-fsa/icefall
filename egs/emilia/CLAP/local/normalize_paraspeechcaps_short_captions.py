import re

import regex


def remove_brackets(text: str) -> str:
    # 删除 (Or) 以及之后的所有内容
    text = re.sub(r"[\(\[\{\<]or[\)\]\}\>].*", "", text, flags=re.I)

    # 去括号及内容
    pattern = re.compile(r"\([^()]*\)|\[[^[\]]*]|\{[^{}]*\}|<[^<>]*>")
    while True:
        new_text = pattern.sub("", text)
        if new_text == text:
            break
        text = new_text

    # 清理残留的单个括号符号
    text = re.sub(r"[()\[\]{}<>]", "", text)

    # 删除 Note: | Or: | Description: 及之后的所有内容
    text = re.sub(r"\b(note|or|description):.*", "", text, flags=re.I)

    # 删除冒号之前所有内容
    text = re.sub(r"^.*?:\s*", "", text)

    return text


def map_phrases(text: str) -> str:
    text = re.sub(r"\bwomen's\b", "woman's", text, flags=re.I)
    text = re.sub(r"\bmen's\b", "man's", text, flags=re.I)
    text = re.sub(r"\bwomen\b", "woman", text, flags=re.I)
    text = re.sub(r"\bmen\b", "man", text, flags=re.I)
    text = re.sub(r"\b([\w-]+)\s+in\s+origin\b", r"\1 accent", text)
    text = re.sub(r"\borigin\b", "accent", text)
    text = re.sub(r"\bcontinent\b", "accent", text)
    text = re.sub(r"\baccents\b", "accent", text)
    text = re.sub(r"aussie", "Australian", text, flags=re.I)
    text = text.replace("environs", "environment")
    text = re.sub(r"\s*,?\s*however\s*,?\s*", " ", text, flags=re.I)

    text = re.sub(r"\s+([,.;!?])", r"\1", text)  # 标点前空格
    text = re.sub(r"\s{2,}", " ", text)  # 连续空格
    text = re.sub(r"^[,.;!?]+\s*", "", text)  # 标点开头
    text = text.replace(",,", ",")
    text = text.replace(",.", ".")
    text = text.replace(".,", ".")
    text = text.replace("..", ".")
    text = text.strip()

    # 修正每个句首大小写
    text = re.sub(
        r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
    )

    return text


def process_accent(text: str, accent: str) -> str:
    if len(accent) == 0:
        return text

    def to_display_form(w: str) -> str:
        w = w.lower()
        parts = re.split(r"([/\-\s])", w)
        return "".join(p.capitalize() if p.isalpha() else p for p in parts).strip()

    if "/" in accent:
        accent = [accent] + accent.split("/")
    else:
        accent = [accent]

    is_missing = True
    for w in accent:

        base = w.lower()
        display = to_display_form(w)

        exact_pattern = re.compile(rf"\b{re.escape(base)}\b", re.I)
        m = exact_pattern.search(text)

        if not m:
            max_edit = min(2, max(0, len(w.replace(" ", "")) - 5))
            fuzzy_pattern = regex.compile(
                rf"(?i)\b({regex.escape(base)}){{e<={max_edit}}}\b"
            )
            m = fuzzy_pattern.search(text)

        if m:
            matched_text = m.group()

            if " " not in base and " " in matched_text.strip():
                continue

            span = m.span()
            prefix = " " if matched_text.startswith(" ") else ""
            suffix = " " if matched_text.endswith(" ") else ""
            text = text[: span[0]] + prefix + display + suffix + text[span[1] :]

            is_missing = False

    if is_missing:
        text, count = re.subn(r"(?i)\b\w+\s+accent\b", f"{display} accent", text)
        if count == 0:
            text += f" {display} accent."

    return text


def normalize(text: str, accent: str) -> str:
    text = remove_brackets(text)
    text = map_phrases(text)
    text = process_accent(text, accent)
    return text


def _normalize(ori_text: str, accent: str) -> tuple[str, str]:
    norm_text = remove_brackets(ori_text)
    norm_text = map_phrases(norm_text)
    norm_text = process_accent(norm_text, accent)
    return ori_text, norm_text


if __name__ == "__main__":
    import difflib
    import sys
    from multiprocessing import Pool

    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    def color_diff_ori(ori_text: str, norm_text: str) -> str:
        sm = difflib.SequenceMatcher(a=ori_text, b=norm_text, autojunk=False)
        out = []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            # tag: 'equal', 'replace', 'delete', 'insert'
            if tag == "equal":
                out.append(ori_text[i1:i2])
            elif tag in ("replace", "delete"):
                out.append(f"{RED}{ori_text[i1:i2]}{RESET}")
            elif tag == "insert":
                out.append(f"{GREEN}{norm_text[j1:j2]}{RESET}")

        return "".join(out)

    input_path = sys.argv[1]
    success_path = sys.argv[2]
    badcase_path = sys.argv[3]

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n").rsplit("    ", 1) for line in f]

    with Pool(processes=64) as pool:
        normalized = pool.starmap(_normalize, lines)

    with Pool(processes=64) as pool:
        colored = pool.starmap(color_diff_ori, normalized)

    with open(success_path, "w", encoding="utf-8") as f_success, open(
        badcase_path, "w", encoding="utf-8"
    ) as f_bad:

        for (ori_text, norm_text), diff_line in zip(normalized, colored):
            if ori_text != norm_text:
                f_success.write(diff_line + "\n")
                # f_success.write(norm_text + "\n")
            else:
                f_bad.write(ori_text + "\n")
