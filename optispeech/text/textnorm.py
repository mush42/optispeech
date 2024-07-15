import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str, *, normalize: bool=True) -> str:
    if normalize:
        text = unicodedata.normalize("NFKC", text)
    text = collapse_whitespace(text)
    return text


def collapse_whitespace(text):
    text = re.sub(WHITESPACE_RE, " ", text)
    return text




