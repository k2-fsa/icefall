"""Same English word normalization for LibriSpeech refs and hypotheses."""
import re


def asr_text_post_processing(text):
    return " ".join(re.sub(r"[^A-Z0-9' ]", " ", text.upper()).split())
