"""Tokenizer Module."""

from typing import Callable, List

import stanza
from nltk.tokenize import word_tokenize
from sacremoses import MosesTokenizer


class Tokenizer:
    """Abstract class to Tokenizers."""

    def forward(self, text: str) -> List[str]:
        """Abstract class for Tokenizers."""
        pass


class TokenizerNLTK(Tokenizer):
    """Tokenizer using NLTK basic version."""

    def __init__(self, language: str = "portuguese") -> None:
        """Tokenizer initialization.

        Args:
            language (str): text language, Default: "portuguese"
        """
        self.language = language

    def forward(self, text: str) -> List[str]:
        """Tokenize the document.

        Args:
            text (str): text to be tokenized.

        Returns: List[str]
        """
        return word_tokenize(text, language=self.language)


class TokenizerStanza(Tokenizer):
    """Stanza tokenizer."""

    def __init__(self, language: str = "pt", token_attr: str = "text", *args, **kwargs) -> None:
        """Stanza tokenizer.

        Args:
            language (str): text language, Default: "pt"
            token_attr (str): what to be returned as token, default: "text", can also be "lemma"
        """
        stanza.download(language)
        self.nlp = stanza.Pipeline(language, *args, **kwargs)
        self.token_attr = token_attr

    def forward(self, tokens: str) -> List[str]:
        """Tokenizer.

        Args:
            tokens (str): List of tokens to be normalized.

        Returns: List[str]
        """
        return [getattr(token, self.token_attr) for sentence in self.nlp(tokens).sentences for token in sentence.words]


class TokenizerMoses(Tokenizer):
    """Moses tokenizer."""

    def __init__(self, language: str = "pt") -> None:
        """Moses tokenizer.

        Args:
            language (str): text language, Default: "pt"
        """
        self.tokenizer = MosesTokenizer(lang=language)

    def forward(self, tokens: str) -> List[str]:
        """Tokenizer.

        Args:
            tokens (str): List of tokens to be normalized.

        Returns: List[str]
        """
        return self.tokenizer.tokenize(tokens)


class TokenizerLambda(Tokenizer):
    """Lambda tokenizer."""

    def __init__(self, lambda_function: Callable) -> None:
        """Lambda tokenizer.

        Args:
            lambda (str): text language, Default: "pt"
        """
        self.tokenizer = lambda_function

    def forward(self, tokens: str) -> List[str]:
        """Tokenizer.

        Args:
            tokens (str): List of tokens to be normalized.

        Returns: List[str]
        """
        return self.tokenizer(tokens)
