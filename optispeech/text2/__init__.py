import os
# if os.environ.get("version","v1")=="v1":
#   from text.symbols import symbols
# else:
#   from text.symbols2 import symbols
import LangSegment
from optispeech.text2.cleaner import clean_text
LangSegment.setfilters(["zh","en"])

from optispeech.text2 import symbols as symbols_v1
from optispeech.text2 import symbols2 as symbols_v2

_symbol_to_id_v1 = {s: i for i, s in enumerate(symbols_v1.symbols)}
_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}

def cleaned_text_to_sequence(cleaned_text, version=None):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  if version is None:version=os.environ.get('version', 'v2')
  if version == "v1":
    phones = [_symbol_to_id_v1[symbol] for symbol in cleaned_text]
  else:
    phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]

  return phones

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


class MixTokenizer:
  def __init__(self, version=None, add_blank=None, add_bos_eos=None, normalize_text=None):
    if version is None:version=os.environ.get('version', 'v1')
    if version == "v1":
      from optispeech.text2 import chinese, japanese, english
      self.language_module_map = {"zh": chinese, "ja": japanese, "en": english}
    else:
      from optispeech.text2 import chinese2, japanese, english, korean, cantonese
      self.language_module_map = {"zh": chinese2, "ja": japanese, "en": english, "ko": korean, "yue": cantonese}

  def __call__(self, text, language, split_sentences: bool = False):
      text_segments = []
      for segment in LangSegment.getTexts(text):
          cleaned_text, _, _ = clean_text(segment["text"], segment["lang"])
          text_segments.append(cleaned_text)
      phones = sum(text_segments, [])
      text_id = cleaned_text_to_sequence(phones)
      # add blank token between segments
      text_id = intersperse(text_id, 0)
      return text_id, text
