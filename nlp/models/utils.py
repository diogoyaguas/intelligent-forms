"""Utilities for models."""

from collections import Counter

import torch
from torchtext.vocab import Vocab
from gensim.models import KeyedVectors
from tqdm import tqdm


def gensim_embeddings(embeddings_path: str) -> Vocab:
    """Convert gensim pre-trained embeddings to pytorch.

    Args:
        embeddings_path (str): Gensim pretrained embeddings.

    Returns: Vocab
    """
    model = KeyedVectors.load_word2vec_format(embeddings_path)
    vocab = model.vocab
    counter: Counter = Counter()

    embedding_dim = model.vectors[0].size
    word2vec_vectors = [torch.zeros(embedding_dim)] * 3
    for token, item in tqdm(vocab.items()):
        word2vec_vectors.append(torch.from_numpy(model[token].copy()))
        counter[token] = item.count
    specials = ('<unk>', '<pad>', '<eos>')
    torch_vocab = Vocab(counter, specials=specials)

    torch_vocab.set_vectors(torch_vocab.stoi, word2vec_vectors, embedding_dim)
    return torch_vocab
