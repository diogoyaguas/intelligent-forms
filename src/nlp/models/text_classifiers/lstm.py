"""CNN Text Module."""

from typing import Any, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from nlp.models.text_classifiers.text_classification import TextClassification


class LSTMModel(TextClassification):
    """based on https://arxiv.org/pdf/1408.5882.pdf but using a LSTM instead of a CNN to extract features."""

    def __init__(self, num_class: int, embeddings: int = None, vocab: Vocab = None, num_layers: int = 1,
                 hidden_size: int = 70, dropout: float = 0.5, params: dict = None, loss: Any = None
                 ) -> None:
        """Initialize model.

        Args:
            num_class (int): number of classes.
            embeddings (int): if not used pretrained embeddings, then will this value as number of embeddings.
            vocab (Vocab): torch.Vocab
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout value.
            hidden_size (int): Hidden size of the LSTM.
        """
        super().__init__(num_class, params, loss)
        self.embeddings = embeddings
        if vocab:
            self.vocab = vocab
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_value = dropout

        self.setup_model()
        self.params["vocab"] = len(self.vocab)
        self.params["embeddings_dim"] = self.embedding.embedding_dim
        self.setup_logs()
        self.save_hyperparameters("num_class", "num_layers", "hidden_size", "dropout", "params")

    def setup_model(self):
        """Start the model weights."""
        if isinstance(self.embeddings, int):
            self.embedding = nn.Embedding(len(self.vocab), self.embeddings, padding_idx=1)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.vocab.vectors), padding_idx=1)

        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim, num_layers=self.num_layers,
                            hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(self.hidden_size * 2, self.num_class)

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, text):
        """Model forward pass.

        Args:
            text (torch.Tensor): text given to the model for the forward pass.

        Returns: torch.Tensor
        """
        x = self.embedding(text)
        lstm, _ = self.lstm(x)
        lstm = lstm.permute(0, 2, 1)
        pooled = self.dropout(F.max_pool1d(lstm, lstm.shape[2])).squeeze(2)

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
