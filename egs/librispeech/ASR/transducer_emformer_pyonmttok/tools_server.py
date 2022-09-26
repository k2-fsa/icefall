import difflib
import re
import string
from collections import defaultdict


def remove_punct(word):
    return word.lower().translate(str.maketrans("", "", string.punctuation))


def find_overlap_index(next_sent, already_sent_text):
    matches = difflib.SequenceMatcher(
        None, already_sent_text.lower(), next_sent.lower(), autojunk=False
    ).get_matching_blocks()
    ok_matches = [
        m for m in matches if len(already_sent_text) - (m.a + m.size) < 4
    ]
    max_match = ok_matches[-2] if len(ok_matches) > 1 else matches[-1]
    # print(
    #     max_match,
    #     matches,
    #     already_sent_text.split()[-10:],
    #     next_sent.split()[-10:],
    # )
    if max_match.size == 0:
        return len(already_sent_text)
    return (max_match.b + max_match.size) + (
        len(already_sent_text) - (max_match.a + max_match.size)
    )


def get_word_and_punct_item(word, index, icefall_output_split):
    word_and_punct = word
    has_punct = not remove_punct(word[-1])
    if index + 1 < len(icefall_output_split) and not remove_punct(
        icefall_output_split[index + 1]
    ):
        word_and_punct += " " + icefall_output_split[index + 1]
        has_punct = True
    word_and_punct_before = word_and_punct
    if index > 0 and not remove_punct(icefall_output_split[index - 1][-1]):
        word_and_punct_before = (
            icefall_output_split[index - 1][-1] + " " + word_and_punct_before
        )
    return {
        "index": index,
        "word_and_punct": word_and_punct,
        "word_and_punct_before": word_and_punct_before,
        "has_punct": has_punct,
    }


def find_match_and_merge_punct(
    word, prev_punct, last_index, icefall_output_words_no_punct
):
    word_punct_maybe = word
    if not icefall_output_words_no_punct[word]:
        prev_punct = False
    for elem in icefall_output_words_no_punct[word]:
        if last_index < elem["index"] <= last_index + 4 or (
            len(icefall_output_words_no_punct[word]) == 1
        ):
            last_index = elem["index"]
            if not prev_punct:
                word_punct_maybe = elem["word_and_punct_before"]
            else:
                word_punct_maybe = elem["word_and_punct"]
            prev_punct = elem["has_punct"]
            break
    return word_punct_maybe, prev_punct, last_index


def get_icefall_punct_items(icefall_output):
    icefall_output_split = icefall_output.split()
    icefall_output_words_no_punct = defaultdict(list)
    for index, word_icefall in enumerate(icefall_output_split):
        word_and_punct_item = get_word_and_punct_item(
            word_icefall, index, icefall_output_split
        )
        icefall_output_words_no_punct[remove_punct(word_icefall)].append(
            word_and_punct_item
        )
    return icefall_output_words_no_punct


def compute_azure_punct(azure_output, icefall_output_words_no_punct):
    azure_output_punct = []
    last_index = 0
    prev_punct = False
    for word_azure in azure_output.split():
        word_punct_maybe, prev_punct, last_index = find_match_and_merge_punct(
            word_azure, prev_punct, last_index, icefall_output_words_no_punct
        )
        azure_output_punct.append(word_punct_maybe)
    azure_punct = " ".join(azure_output_punct)
    azure_punct = azure_punct.replace(" ,", ",").replace(" .", ".")

    # remove last punctuation, it will be added during next merge
    # not necessary ?
    # if azure_punct and not remove_punct(azure_punct[-1]):
    #     azure_punct = azure_punct[:-1]
    return azure_punct


def merge_punct(azure_output, icefall_output, max_words=None):
    if max_words:
        azure_output = " ".join(azure_output.split()[-max_words:])
        icefall_output = " ".join(icefall_output.split()[-max_words:])

    icefall_output_words_no_punct = get_icefall_punct_items(icefall_output)

    return compute_azure_punct(azure_output, icefall_output_words_no_punct)


def fix_punct(text):
    return re.sub(
        "([.?!])\s*([a-zA-Z])",
        lambda p: p.group(0).upper(),
        text,
    )


def clean_icefall_text(text):
    return re.sub(
        r"｟(\d)_\d｠",
        "\\1",
        text.replace("｟speaker_change｠", "").replace(
            "｟maybe_speaker_change｠", ""
        ),
    )


def remove_short_sentences(text):
    return re.sub(r"\. [\w| ]{0,13}\b(?<!\b[m|M]erci)(\.)", "\\1", text)


def lower_list(ls):
    return [l.lower() for l in ls]


def remove_interjections(text):
    interjections = {"euh", "ben", "hein", "voilà", "bon", "ah"}
    for inter in interjections:
        text = re.sub(f"( ){inter}( |.)", "\\2", text, flags=re.IGNORECASE)
    return text


def remove_repetitions(already_sent_text, no_overlap_azure_text):
    already_sent_text_split = already_sent_text.split()[-3:]
    no_overlap_azure_text_split = no_overlap_azure_text.split()

    # 1-2-3-gram
    for ngram in [3, 2, 1]:
        new_no_overlap_azure_text_split = []
        sent1 = (
            already_sent_text_split[-ngram:]
            + no_overlap_azure_text_split[:-ngram]
        )
        sent2 = no_overlap_azure_text_split
        for i in range(len(sent2) - (ngram - 1)):
            if lower_list(sent1[i : i + ngram]) != lower_list(
                sent2[i : i + ngram]
            ):
                new_no_overlap_azure_text_split.append(sent2[i])
        if ngram > 1:
            new_no_overlap_azure_text_split += no_overlap_azure_text_split[
                -(ngram - 1) :
            ]
        no_overlap_azure_text_split = new_no_overlap_azure_text_split

    maybe_start_space = (
        " "
        if no_overlap_azure_text and no_overlap_azure_text[0] == " "
        else ""
    )
    maybe_end_space = (
        " "
        if no_overlap_azure_text
        and no_overlap_azure_text[-1] == " "
        and len(no_overlap_azure_text) > 1
        else ""
    )
    return (
        maybe_start_space
        + " ".join(no_overlap_azure_text_split)
        + maybe_end_space
    )


def get_text_ready_for_submission(azure_text, icefall_text, already_sent_text):
    icefall_formatted_text = clean_icefall_text(icefall_text)
    azure_text_with_punct = merge_punct(
        azure_text,
        icefall_formatted_text,
        max_words=20,
    )

    # remove last word in case not finished. Contributes to latency
    azure_text_with_punct = azure_text_with_punct.rsplit(" ", 1)[0]

    azure_text_with_punct = fix_punct(azure_text_with_punct)
    overlap_index = find_overlap_index(
        azure_text_with_punct, already_sent_text
    )
    no_overlap_azure_text = azure_text_with_punct[overlap_index:]
    print("no overlap ", no_overlap_azure_text + "|")
    print("no overlap 2", azure_text_with_punct[:overlap_index] + "|")

    no_repetitions_no_overlap_azure_text = remove_repetitions(
        already_sent_text, no_overlap_azure_text
    )
    print("no overlap 3", no_repetitions_no_overlap_azure_text + "|")
    no_repetitions_no_overlap_azure_text = remove_short_sentences(
        no_repetitions_no_overlap_azure_text
    )
    print("no overlap 4", no_repetitions_no_overlap_azure_text + "|")

    no_repetitions_no_overlap_azure_text = remove_interjections(
        no_repetitions_no_overlap_azure_text
    )
    return no_repetitions_no_overlap_azure_text
