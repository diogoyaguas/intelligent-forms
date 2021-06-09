"""Bert Text Module."""

from typing import Any

import torch
from torch import nn, optim
from transformers import BertTokenizer, BertModel

from nlp.models.text_classifiers.text_classification import TextClassification


class BertTextClassifier(TextClassification):
    """based on https://arxiv.org/abs/1905.05583 ."""

    def __init__(self, num_class: int, bert_version: str = 'bert-base-multilingual-cased', params: dict = None,
                 loss: Any = None) -> None:
        """Initialize model.

        Args:
            num_class (int): number of classes.
            bert_version (str): bert version from huggingface.com
        """
        super().__init__(num_class, params, loss)
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)
        self.model = BertModel.from_pretrained(bert_version)
        self.model.pooler = None
        self.classifier = nn.Linear(768, num_class)
        self.setup_logs()
        self.save_hyperparameters("num_class", "bert_version", "params")

    def forward(self, text):
        """Model forward pass.

        Args:
            text (torch.Tensor): text given to the model for the forward pass.

        Returns: torch.Tensor
        """
        input_ids, attention_mask = text
        input_ids = input_ids.to(device="cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = attention_mask.to(device="cuda" if torch.cuda.is_available() else "cpu")
        x = self.model(input_ids, attention_mask)

        x = x["last_hidden_state"][:, 0, :]

        return torch.tanh(self.classifier(x))

    def generate_batch(self, batch):
        """Generate the batch to give to the model.

        Args:
            batch (Any): Text and labels

        Returns: Tuple[torch.Tensor, torch.Tensor]
        """
        text = [entry[0] for entry in batch]
        label = torch.tensor([entry[1] for entry in batch], dtype=torch.long)

        text = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        return [text['input_ids'], text['attention_mask']], label

    def configure_optimizers(self):
        """Configure optimizer for pytorch lighting.

        Returns: optimizer for pytorch lighting.

        """
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=0.00001, steps_per_epoch=13822, epochs=10, pct_start=0.1),
            'interval': 'step',
        }

        return [optimizer], [scheduler]
