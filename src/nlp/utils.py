"""Utilities Module."""

from collections import Counter
from typing import Any

from tqdm import tqdm
from torchtext.vocab import Vocab


def build_vocab_from_iterator(iterator: Any, **kwargs) -> Vocab:
    """Build torchtext Vocab.

    Args:
        iterator (Iterator): iterate a dataset

    Returns: Vocab
    """
    counter: Counter = Counter()
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = Vocab(counter, specials=('<unk>', '<pad>', '<eos>'), **kwargs)
    return word_vocab
