from typing import Any

from .normalization import UNICODE_NORM_FORM
from .tokenizers import BaseTokenizer


class TextProcessor:
    def __init__(
        self, tokenizer: str|Any, add_blank: str, add_bos_eos: str, normalize_text: bool, languages: list[str]
    ):
        self.tokenizer_ref = tokenizer
        self.add_blank = add_blank
        self.add_bos_eos = add_bos_eos
        self.normalize_text = normalize_text
        self.languages = languages
        if isinstance(self.tokenizer_ref, str):
            tokenizer_cls = BaseTokenizer.get_tokenizer_by_name(self.tokenizer_ref)
        else:
            # The user passed a class which is partially initialized by hydra
            tokenizer_cls = self.tokenizer_ref
        self.tokenizer = tokenizer_cls(add_blank=add_blank, add_bos_eos=add_bos_eos, normalize_text=normalize_text)
        self.num_languages = len(languages)
        self.is_multi_language = self.num_languages > 1
        self.default_language = languages[0].strip().lower()

    def __call__(self, text, lang, split_sentences: bool = False):
        # handle special value
        if lang is None:
            lang = self.default_language
        lang = lang.strip().lower()
        if lang not in self.languages:
            raise ValueError(f"Language {lang} does not exist in the supported language list.")
        return self.tokenizer(text, language=lang, split_sentences=split_sentences)

    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)

    def asdict(self):
        return dict(
            tokenizer_name=self.tokenizer_name,
            add_blank=self.add_blank,
            add_bos_eos=self.add_bos_eos,
            normalize_text=self.normalize_text,
            languages=self.languages,
        )
