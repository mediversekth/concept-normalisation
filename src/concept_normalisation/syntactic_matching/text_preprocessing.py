import re
import string

from concept_normalisation.config import (
    PREPROCESS_LOWERCASE, PREPROCESS_REMOVE_PUNCTUATION, PREPROCESS_REMOVE_STOPWORDS,
)

DEFAULT_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with",
    "is", "are", "was", "were", "by", "at", "as", "from",
}


class TextPreprocessor:
    def __init__(
        self,
        lowercase: bool = PREPROCESS_LOWERCASE,
        remove_punctuation: bool = PREPROCESS_REMOVE_PUNCTUATION,
        remove_stopwords: bool = PREPROCESS_REMOVE_STOPWORDS,
        stopwords: set = None,
        strip_whitespace: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.strip_whitespace = strip_whitespace
        self._punct_table = str.maketrans("", "", string.punctuation)

    def clean(self, text) -> str:
        if text is None:
            return ""
        text = str(text)

        if self.strip_whitespace:
            text = text.strip()
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = text.translate(self._punct_table)
        if self.remove_stopwords:
            text = " ".join(w for w in text.split() if w not in self.stopwords)

        return re.sub(r"\s+", " ", text).strip()

    def clean_series(self, series) -> list:
        return [self.clean(text) for text in series]
