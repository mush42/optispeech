SYMBOLS = [
    "_",
    "^",
    "$",
    " ",
    "!",
    '"',
    "#",
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
    "ⱱ",
]


# Special symbols
PAD = "_"
BOS = "^"
EOS = "$"

# Special symbol ids
PAD_ID = SYMBOLS.index(PAD)
BOS_ID = SYMBOLS.index(BOS)
EOS_ID = SYMBOLS.index(EOS)
SPACE_ID = SYMBOLS.index(" ")

# Mappings from symbol to numeric ID and vice versa:
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}  # pylint: disable=unnecessary-comprehension


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
