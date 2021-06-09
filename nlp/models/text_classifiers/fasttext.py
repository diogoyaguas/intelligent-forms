"""Fast Text Module."""

from typing import Any, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from nlp.models.text_classifiers.text_classification import TextClassification


class FastText(TextClassification):
    """based on https://arxiv.org/abs/1607.01759 ."""

    def __init__(self, num_class: int, embeddings: int = 10, vocab: Vocab = None, params: dict = None, loss: Any = None
                 ) -> None:
        """Initialize model.

        Args:
            num_class (int): number of classes.
            embeddings (int): if not used pretrained embeddings, then will this value as number of embeddings.
            vocab (Vocab): torch.Vocab
        """
        super().__init__(num_class, params, loss)
        self.embeddings = embeddings
        if vocab:
            self.vocab = vocab

        self.setup_model()
        self.params["vocab"] = len(self.vocab)
        self.params["embeddings_dim"] = self.embedding.embedding_dim
        self.setup_logs()
        self.save_hyperparameters("num_class", "params")

    def setup_model(self):
        """Start the model weights."""
        if isinstance(self.embeddings, int):
            self.embedding = nn.Embedding(len(self.vocab), self.embeddings, padding_idx=1)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.vocab.vectors), padding_idx=1)

        self.fc = nn.Linear(self.embedding.embedding_dim, self.num_class)

    def forward(self, text):
        """Model forward pass.

        Args:
            text (torch.Tensor): text given to the model for the forward pass.

        Returns: torch.Tensor
        """
        embedded = self.embedding(text)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)

    def generate_batch(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the batch to give to the model.

        Args:
            batch (Any): Text and labels

        Returns: Tuple[torch.Tensor, torch.Tensor]
        """
        text = [[self.vocab[token] for token in entry[0]] for entry in batch]
        text = [document.append(2) or torch.tensor(document, dtype=torch.long) for document in text]  # type: ignore
        text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=1)

        label = torch.tensor([entry[1] for entry in batch], dtype=torch.long)

        return text, label
