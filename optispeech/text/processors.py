# coding: utf-8

from .symbols import BOS_ID, EOS_ID, ID_TO_SYMBOL, PAD_ID, SYMBOL_TO_ID


def phonemes_to_ids(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    for symbol in text:
        symbol_id = SYMBOL_TO_ID[symbol]
        sequence.append(symbol_id)
    return sequence


def ids_to_phonemes(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = ID_TO_SYMBOL[symbol_id]
        result += s
    return result


def phonemes_to_ids_piper(phonemes):
    id_map = SYMBOL_TO_ID
    ids = [BOS_ID]
    for phoneme in phonemes:
        ids.append(id_map[phoneme])
        ids.append(PAD_ID)
    ids.append(EOS_ID)
    return ids


def ids_to_phonemes_piper(ids):
    return "".join([ID_TO_SYMBOL[id] for id in ids])
