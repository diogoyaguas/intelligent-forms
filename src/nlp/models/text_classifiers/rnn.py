"""RNN Text Module."""

from typing import Any, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from nlp.models.text_classifiers.text_classification import TextClassification


class GruTextClassification(TextClassification):
    """Simple GRU architecture."""

    def __init__(self, num_class, embeddings: int = None, vocab: Vocab = None, n_layers: int = 1, dropout: float = 0.5,
                 params: dict = None, loss: Any = None) -> None:
        """Initialize model.

        Args:
            num_class (int): number of classes.
            embeddings (int): if not used pretrained embeddings, then will this value as number of embeddings.
            vocab (Vocab): torch.Vocab
            n_layers (int): Number of GRU layers.
            dropout (float): Dropout value.
        """
        super().__init__(num_class, params, loss)
        self.embeddings = embeddings
        if vocab:
            self.vocab = vocab
        self.n_layers = n_layers
        self.dropout_value = dropout

        self.setup_model()
        self.params["vocab"] = len(self.vocab)
        self.params["embeddings_dim"] = self.embedding.embedding_dim
        self.setup_logs()
        self.save_hyperparameters("num_class", "n_layers", "dropout", "params")

    def setup_model(self):
        """Start the model weights."""
        if isinstance(self.embeddings, int):
            self.embedding = nn.Embedding(len(self.vocab), self.embeddings, padding_idx=1)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.vocab.vectors), padding_idx=1)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.n_filters,
                                              kernel_size=(fs, self.embedding.embedding_dim))
                                    for fs in self.filter_sizes])
        embed_dim = self.embedding.embedding_dim
        self.features = nn.Sequential(*[
            BidirectionalGRU(input_size=embed_dim if i == 0 else embed_dim * 2,
                             hidden_size=embed_dim, dropout=self.dropout)
            for i in range(self.n_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(embed_dim, self.num_class)
        )

    def forward(self, text):
        """Model forward pass.

        Args:
            text (torch.Tensor): text given to the model for the forward pass.

        Returns: torch.Tensor
        """
        x = self.embedding(text)
        x = self.features(x)
        x = x[:, -1, :].squeeze(1)
        return self.classifier(x)

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


class BidirectionalGRU(nn.Module):
    """BidirectionalGRU layer."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        """Create BidirectionalGRU layer.

        Args:
            input_size (int): Input size for the GRU layer.
            hidden_size (int): Number of hidden parameters for the GRU layer.
            dropout (float): Dropout Value.
        """
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Model forward pass.

        Args:
            x (torch.Tensor): x given to the model for the forward pass.

        Returns: torch.Tensor
        """
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x
