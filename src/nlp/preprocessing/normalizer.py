"""Normalizer Module."""
import io
import os
import re
import requests
from typing import List
from string import punctuation
from zipfile import ZipFile

import nltk
import stanza
from hunspell import Hunspell
from nltk.stem.snowball import SnowballStemmer
from nltk.metrics.distance import edit_distance
from unidecode import unidecode


class Normalizer:
    """Abstract class to Normalizers."""

    def forward(self, tokens: List[str]) -> List[str]:
        """Abstract class to Tokenizers."""
        pass


class SnowballStemmerNLTK(Normalizer):
    """NLTK basic steammer."""

    def __init__(self, language: str = "portuguese") -> None:
        """NLTK basic steammer.

        Args:
            language (str): text language, Default: "portuguese"
        """
        self.stemmer = SnowballStemmer(language=language, ignore_stopwords=False)

    def forward(self, tokens: List[str]) -> List[str]:
        """Steam the tokens.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        return [self.stemmer.stem(token) for token in tokens]


class RemoveStopWords(Normalizer):
    """Stop word Remover."""

    def __init__(self, language: str = "portuguese") -> None:
        """Remove Stop words.

        Args:
            language (str): text language, Default: "portuguese"
        """
        self.stop_words = nltk.corpus.stopwords.words(language)
        self.stop_words = [unidecode(token) for token in self.stop_words]

    def forward(self, tokens: List[str]) -> List[str]:
        """Remove Stop words.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        return [token for token in tokens if unidecode(token) not in self.stop_words]


class RemovePunctuation(Normalizer):
    """Remove Punctuation."""

    def forward(self, tokens: List[str]) -> List[str]:
        """Remove punctuation.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        return [token.translate(str.maketrans('', '', punctuation)) for token in tokens
                if len(token.translate(str.maketrans('', '', punctuation))) > 0]


class StanzaLemmatizer(Normalizer):
    """Stanza Lemmatizer."""

    def __init__(self, language: str = "pt") -> None:
        """Stanza lemmatizer.
        Args:
            language (str): text language, Default: "portuguese"
        """
        stanza.download(language)
        self.nlp = stanza.Pipeline(language)

    def forward(self, tokens: List[str]) -> List[str]:
        """Lemmatize Tokens.
        Args:
            tokens (List[str]): List of tokens to be normalized.
        Returns: List[str]
        """
        return [(token if token == ' ' else self.nlp(token).sentences[0].words[0].lemma) for token in tokens]


class Lower(Normalizer):
    """Lower tokens."""

    def forward(self, tokens: List[str]) -> List[str]:
        """Lower Tokens.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        return [token.lower() for token in tokens]


class SpellChecker(Normalizer):
    """Hunspell spell checker."""

    def __init__(self, hunspell_name: str = 'pt_PT-preao',
                 hunspell_data_dir: str = './data/hunspell-pt_PT-preao-20210106',
                 max_distance: int = 3):
        """Load Hunspell Dictionary.

        Args:
            hunspell_name (str): name of the dictionary
            hunspell_data_dir (str): directory where the dictionary is at
        """
        self.max_distance = max_distance
        self.h = Hunspell(hunspell_name, hunspell_data_dir=hunspell_data_dir)

    def forward(self, tokens: List[str]) -> List[str]:
        """Spellchecker.

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """
        tokens_checked = []
        for token in tokens:
            suggestions = self.h.suggest(token)
            if len(suggestions) > 0 and len(token) > 1:
                suggestion = suggestions[0]
                if edit_distance(token, suggestion) <= self.max_distance:
                    tokens_checked.append(suggestion)
            else:
                tokens_checked.append(token)
        return tokens_checked


class SynonymSubstitution(Normalizer):
    """Synonym Substitution."""
    pattern1 = re.compile(r'\d+ : \w+ : (?:.+\(\d+\.\d+\);)+')
    pattern2 = re.compile(r'\s?(.+)\(\d+\.\d+\)')

    def __init__(self, filepath: str = "./data/conto_pt.txt"):

        if not os.path.exists(filepath):
            r = requests.get(url='http://ontopt.dei.uc.pt/recursos/CONTO.PT.01.zip', stream=True)
            with ZipFile(io.BytesIO(r.content)) as archive:
                data = archive.read('contopt_0.1_r2_c0.0.txt')
                f = open(filepath, 'wb')
                f.write(data)
                f.close()
        self.synonyms = SynonymSubstitution._load_synonyms(filepath)

    @staticmethod
    def _load_synonyms(filename):
        f = open(filename, "r", encoding="utf-8")
        d = dict()
        line = f.readline()
        while line != "":
            if SynonymSubstitution.pattern1.match(line):
                synonyms_and_conf = line.split(":")[2]
                synonyms_and_conf = synonyms_and_conf.split(";")
                synonyms_and_conf.pop()
                synonyms = [SynonymSubstitution.pattern2.match(synonym_and_conf).group(1) for synonym_and_conf in synonyms_and_conf]
                best = synonyms[0]
                for i in range(1, len(synonyms)):
                    d[synonyms[i]] = best
            line = f.readline()
        f.close()
        return d

    def forward(self, tokens: List[str]) -> List[str]:
        """Synonym Substitution..

        Args:
            tokens (List[str]): List of tokens to be normalized.

        Returns: List[str]
        """

        return [self.synonyms.get(token, token) for token in tokens]
