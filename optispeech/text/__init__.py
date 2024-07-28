from typing import List, Tuple, Union

from piper_phonemize import phonemize_espeak

from ..utils import intersperse
from . import processors
from . import symbols
from .textnorm import UNICODE_NORM_FORM, collapse_whitespace, preprocess_text


def get_input_symbols():
    special_symbols = dict(
        pad=symbols.PAD,
        bos=symbols.BOS,
        eos=symbols.EOS,
    )
    return symbols.SYMBOL_TO_ID, special_symbols


def process_and_phonemize_text(
    text: str,
    language: str,
    tokenizer='default',
    *,
    add_blank: bool=True,
    add_bos_eos: bool=False,
    normalize_text: bool=True,
    split_sentences: bool=False
) -> Tuple[Union[List[int], List[List[int]]], str]:
    tok = tokenizer.lower()
    if tok == 'default':
        return process_and_phonemize_text_default(text, language, add_blank=add_blank, add_bos_eos=add_bos_eos, normalize=normalize_text, split_sentences=split_sentences)
    elif tokenizer == 'piper':
        return process_and_phonemize_text_piper(text, language, add_blank=add_blank, add_bos_eos=add_bos_eos, normalize=normalize_text, split_sentences=split_sentences)
    raise ValueError(f"Unknown tokenizer `{tokenizer}`")


def process_and_phonemize_text_default(text: str, language: str, *, add_blank: bool, add_bos_eos: bool, normalize: bool=True, split_sentences: bool) -> Tuple[Union[List[int], List[List[int]]], str]:
    phonemes, normalized_text = phonemize_text(text, language, normalize=normalize)
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
        if add_bos_eos:
            phoneme_ids = [
                symbols.BOS_ID,
                *phoneme_ids,
                symbols.EOS_ID,
            ]
    else:
        phoneme_ids = []
        for sent_ph in phonemes:
            sent_phonemes = list(collapse_whitespace("".join(sent_ph)))
            phids = processors.phonemes_to_ids(sent_phonemes)
            if add_blank:
                phids = intersperse(phids, 0)
            if add_bos_eos:
                phids = [symbols.BOS_ID, *phids, symbols.EOS_ID]
            phoneme_ids.append(phids)
    return phoneme_ids, normalized_text


def process_and_phonemize_text_piper(text: str, language: str, *, add_blank: bool, add_bos_eos: bool, normalize: bool=True, split_sentences: bool) -> Tuple[Union[List[int], List[List[int]]], str]:
    phonemes, normalized_text = phonemize_text(text, language, normalize=normalize)
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


def phonemize_text(text: str, language: str, *, normalize: bool=True) -> str:
    # Preprocess
    text = preprocess_text(text, normalize=normalize)
    # Phonemize
    phonemes = phonemize_espeak(text, language)
    return phonemes, text

