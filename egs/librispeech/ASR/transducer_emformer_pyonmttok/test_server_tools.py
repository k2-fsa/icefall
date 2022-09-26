import pytest

from tools_server import merge_punct, remove_repetitions, remove_interjections, remove_short_sentences


def test_merge_punct():
    no_punct_sent = "bonjour à toutes et à tous demain"
    punct_sent = "Bonjour à à, toutes et tous. Demain."
    merge_punct_sent = merge_punct(no_punct_sent, punct_sent)
    assert merge_punct_sent == "Bonjour à, toutes et à tous. Demain."


def test_remove_repetitions():
    prev_sent = "bonjour à"
    next_sent = " à tous"
    result = remove_repetitions(prev_sent, next_sent)
    assert result == " tous"

    next_sent = " à à tous"
    result = remove_repetitions(prev_sent, next_sent)
    assert result == " tous"

    next_sent = "à à tous"
    result = remove_repetitions(prev_sent, next_sent)
    assert result == "tous"

    prev_sent = "bonjour à et à"
    next_sent = " et à tous"
    result = remove_repetitions(prev_sent, next_sent)
    assert result == " tous"

    prev_sent = "bonjour à tous et"
    next_sent = " à tous et toutes"
    result = remove_repetitions(prev_sent, next_sent)
    assert result == " toutes"


def test_remove_interjections():
    text = "bonjour euh à tous"
    assert remove_interjections(text) == "bonjour à tous"

    text = "bonjour Euh à tous"
    assert remove_interjections(text) == "bonjour à tous"


def test_remove_short_sentences():
    text = "club. Donc. Dès"
    assert remove_short_sentences(text) == "club. Dès"

# def test_merge_punct_corner_cases():
