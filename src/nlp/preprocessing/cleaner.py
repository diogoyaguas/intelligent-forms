"""Cleaner Module."""

import re
from string import punctuation

from bs4 import BeautifulSoup
from unidecode import unidecode


class Cleaner:
    """Abstract class to Cleaners."""

    def forward(self, text: str) -> str:
        """Abstract class for Cleaners."""
        pass


class AccentRemover(Cleaner):
    """Remove Accents."""

    def forward(self, text: str) -> str:
        """Remove text accents.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """
        return unidecode(text)


class RemoveUrl(Cleaner):
    """Remove URLs."""

    def forward(self, text: str) -> str:
        """Remove text URLs.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+", "", text)
        return text


class RemoveEmail(Cleaner):
    """Remove Emails."""

    def forward(self, text: str) -> str:
        """Remove text Emails.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """
        return re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", " ", text)


class RemoveNumber(Cleaner):
    """Remove Numbers."""

    def forward(self, text: str) -> str:
        """Remove text Numbers.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """
        return re.sub(r'[0-9]+', ' ', text)


class RemoveHTML(Cleaner):
    """Remove HTML."""

    def forward(self, text: str) -> str:
        """Remove HTML from text.

        Args:
            text (str): text to be cleaned.

        Returns: str

        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()


class RemovePunctuation(Cleaner):
    """Remove Punctuation."""

    def forward(self, text: str) -> str:
        """Remove Punctuation from text.

        Args:
            text (str): text to be cleaned.

        Returns: str

        """
        return text.translate(str.maketrans('', '', punctuation))
