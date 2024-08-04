from functools import partial

from optispeech.text import process_and_phonemize_text


class TextProcessor:
    def __init__(self, languages, add_blank, add_bos_eos):
        self.num_languages = len(languages)
        self.is_multi_language = self.num_languages > 1
        self.default_language = languages[0].code.strip().lower()
        self.funcs = {
            lang.code.strip().lower(): partial(
                process_and_phonemize_text,
                language=lang.code,
                tokenizer=lang.get("tokenizer", "default"),
                normalize_text=lang.get("normalize_text", True),
                add_blank=add_blank,
                add_bos_eos=add_bos_eos,
            )
            for lang in languages
        }

    def __call__(self, text, lang, **kwargs):
        # handle special value
        if lang is None:
            lang = self.default_language
        try:
            func = self.funcs[lang.strip().lower()]
        except KeyError:
            raise LookupError(f"Language {lang} does not exist in the supported language list.")
        return func(text, **kwargs)
