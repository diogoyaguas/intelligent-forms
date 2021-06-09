"""CNN Text Module."""

from typing import Any, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from nlp.models.text_classifiers.text_classification import TextClassification


class CNNModel(TextClassification):
    """based on https://arxiv.org/pdf/1408.5882.pdf ."""

    def __init__(self, num_class: int, embeddings: int = None, vocab: Vocab = None, n_filters: int = 100,
                 dropout: float = 0.5, filter_sizes: Tuple[int, ...] = (2, 3, 4), params: dict = None, loss: Any = None
                 ) -> None:
        """Initialize model.

        Args:
            num_class (int): number of classes.
            embeddings (int): if not used pretrained embeddings, then will this value as number of embeddings.
            vocab (Vocab): torch.Vocab
            n_filters (int): Number of output filters for the convolution layer.
            dropout (float): Dropout value.
            filter_sizes (Tuple[int]): Kernel size filters applied on the convolution layer.
        """
        super().__init__(num_class, params, loss)
        self.embeddings = embeddings
        if vocab:
            self.vocab = vocab
        self.n_filters = n_filters
        self.dropout_value = dropout
        self.filter_sizes = filter_sizes

        self.setup_model()
        self.params["vocab"] = len(self.vocab)
        self.params["embeddings_dim"] = self.embedding.embedding_dim
        self.setup_logs()
        self.save_hyperparameters("num_class", "n_filters", "dropout", "filter_sizes", "params")

    def setup_model(self):
        """Start the model weights."""
        if isinstance(self.embeddings, int):
            self.embedding = nn.Embedding(len(self.vocab), self.embeddings, padding_idx=1)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.vocab.vectors), padding_idx=1)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.n_filters,
                                              kernel_size=(fs, self.embedding.embedding_dim))
                                    for fs in self.filter_sizes])

        self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters, self.num_class)

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, text):
        """Model forward pass.

        Args:
            text (torch.Tensor): text given to the model for the forward pass.

        Returns: torch.Tensor
        """
        x = self.embedding(text)
        x = x.unsqueeze(1)
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

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
