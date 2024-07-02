import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = collapse_whitespace(text)
    return text


def collapse_whitespace(text):
    text = re.sub(WHITESPACE_RE, " ", text)
    return text




