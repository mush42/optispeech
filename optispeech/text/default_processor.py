# coding: utf-8

_pad = "_"
_symbols_ipa = [
    " ",
    "!",
    "\"",
    "#",
    "$",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "?",
    "X",
    "^",
    "_",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "æ",
    "ç",
    "ð",
    "ø",
    "ħ",
    "ŋ",
    "œ",
    "ǀ",
    "ǁ",
    "ǂ",
    "ǃ",
    "ɐ",
    "ɑ",
    "ɒ",
    "ɓ",
    "ɔ",
    "ɕ",
    "ɖ",
    "ɗ",
    "ɘ",
    "ə",
    "ɚ",
    "ɛ",
    "ɜ",
    "ɞ",
    "ɟ",
    "ɠ",
    "ɡ",
    "ɢ",
    "ɣ",
    "ɤ",
    "ɥ",
    "ɦ",
    "ɧ",
    "ɨ",
    "ɪ",
    "ɫ",
    "ɬ",
    "ɭ",
    "ɮ",
    "ɯ",
    "ɰ",
    "ɱ",
    "ɲ",
    "ɳ",
    "ɴ",
    "ɵ",
    "ɶ",
    "ɸ",
    "ɹ",
    "ɺ",
    "ɻ",
    "ɽ",
    "ɾ",
    "ʀ",
    "ʁ",
    "ʂ",
    "ʃ",
    "ʄ",
    "ʈ",
    "ʉ",
    "ʊ",
    "ʋ",
    "ʌ",
    "ʍ",
    "ʎ",
    "ʏ",
    "ʐ",
    "ʑ",
    "ʒ",
    "ʔ",
    "ʕ",
    "ʘ",
    "ʙ",
    "ʛ",
    "ʜ",
    "ʝ",
    "ʟ",
    "ʡ",
    "ʢ",
    "ʦ",
    "ʰ",
    "ʲ",
    "ˈ",
    "ˌ",
    "ː",
    "ˑ",
    "˞",
    "ˤ",
    "̃",
    "̊",
    "̝",
    "̧",
    "̩",
    "̪",
    "̯",
    "̺",
    "̻",
    "β",
    "ε",
    "θ",
    "χ",
    "ᵻ",
    "↑",
    "↓",
    "ⱱ"
]


# Export all symbols:
symbols = [_pad] + list(_symbols_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension


def text_to_sequence(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
        
    for symbol in text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result

