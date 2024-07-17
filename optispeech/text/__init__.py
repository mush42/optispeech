from typing import List, Tuple, Union

from piper_phonemize import phonemize_espeak

from ..utils import intersperse
from . import processors
from .textnorm import collapse_whitespace, preprocess_text


def process_and_phonemize_text(
    text: str,
    lang: str,
    tokenizer='default',
    *,
    add_blank: bool=True,
    normalize: bool=True,
    split_sentences: bool=False
) -> Tuple[Union[List[int], List[List[int]]], str]:
    tok = tokenizer.lower()
    if tok == 'default':
        return process_and_phonemize_text_default(text, lang, add_blank=add_blank, normalize=normalize, split_sentences=split_sentences)
    elif tokenizer == 'piper':
        return process_and_phonemize_text_piper(text, lang, add_blank=add_blank, normalize=normalize, split_sentences=split_sentences)
    raise ValueError(f"Unknown tokenizer `{tokenizer}`")


def process_and_phonemize_text_default(text: str, lang: str, *, add_blank: bool, normalize: bool=True, split_sentences: bool) -> Tuple[Union[List[int], List[List[int]]], str]:
    phonemes, normalized_text = phonemize_text(text, lang, normalize=normalize)
    if not split_sentences:
        phonemes = [
            phoneme
            for sentence_phonemes in phonemes
            for phoneme in sentence_phonemes
        ]
        phonemes = list(collapse_whitespace("".join(phonemes)))
        phoneme_ids = processors.phonemes_to_ids(phonemes)
        if add_blank:
            phoneme_ids = intersperse(phoneme_ids, 0)
    else:
        phoneme_ids = []
        for sent_ph in phonemes:
            sent_phonemes = list(collapse_whitespace("".join(sent_ph)))
            phids = processors.phonemes_to_ids(sent_phonemes)
            if add_blank:
                phids = intersperse(phids, 0)
            phoneme_ids.append(phids)
    return phoneme_ids, normalized_text


def process_and_phonemize_text_piper(text: str, lang: str, *, add_blank: bool, normalize: bool=True, split_sentences: bool) -> Tuple[Union[List[int], List[List[int]]], str]:
    phonemes, normalized_text = phonemize_text(text, lang, normalize=normalize)
    if not split_sentences:
        phonemes = [
            phoneme
            for sentence_phonemes in phonemes
            for phoneme in sentence_phonemes
        ]
        phoneme_ids = processors.phonemes_to_ids_piper(phonemes)
    else:
        phoneme_ids = [processors.phonemes_to_ids_piper(ph) for ph in phonemes]
    return phoneme_ids, normalized_text


def phonemize_text(text: str, lang: str, *, normalize: bool=True) -> str:
    # Preprocess
    text = preprocess_text(text, normalize=normalize)
    # Phonemize
    phonemes = phonemize_espeak(text, lang)
    return phonemes, text

