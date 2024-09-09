import warnings

from optispeech.text.tokenizers import BaseTokenizer

from .tokenization import arabic_to_tokens, tokens_to_ids, phon_to_id_
from . import symbols


class ArabicTokenizer(BaseTokenizer):
    name = "arabic-buck"
    input_symbols = dict(phon_to_id_)
    special_symbols = dict(
        pad=phon_to_id_[symbols.PADDING_TOKEN],
        bos=None,
        eos=phon_to_id_[symbols.EOS_TOKEN],
    )

    def __call__(
        self, text: str, language: str, *, split_sentences: bool = True
    ) -> tuple[list[int] | list[list[int]], str]:
        """
        No support for numbers/dates/abbreviation-expansion..etc.
        No support for sentence splitting for now.
        """
        if split_sentences == True:
            warnings.warn("Arabic tokenizer does not support sentence splitting for now.")
        toks = arabic_to_tokens(text)
        ids = tokens_to_ids(toks)
        return ids, text
