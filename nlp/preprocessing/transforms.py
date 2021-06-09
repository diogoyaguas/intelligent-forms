"""Transforms Module."""

from typing import List

from torchtext.data.utils import ngrams_iterator

from nlp.preprocessing.normalizer import Normalizer


class GenerateNgrams(Normalizer):
    """Generate ngrams."""

    def __init__(self, ngrams: int = 2):
        """Select the number the ngrams.

        Args:
            ngrams (int): number of ngrams, cumulative value, if 2 then will be generated 1 and 2 ngrams.
        """
        self.ngrams = ngrams

    def forward(self, tokens: List[str]) -> List[str]:
        """Generate N-grams.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        return list(ngrams_iterator(tokens, self.ngrams))


class TokensToString(Normalizer):
    """Convert Tokens to string."""

    def forward(self, tokens: List[str]) -> str:  # type: ignore
        """Tokens to String.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: str
        """
        return " ".join(tokens)
