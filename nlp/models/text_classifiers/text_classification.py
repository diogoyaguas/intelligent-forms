"""Abstract Module for text classification basic models."""

from typing import Any, Optional

import pytorch_lightning as pl
from torch import optim, nn


class TextClassification(pl.LightningModule):
    """Text Classification Abstract Class."""

    def __init__(self, num_class: int, params: Optional[dict], loss: Any) -> None:
        """Initialize Text Classification.

        Args:
            num_class (int): number of classes.
        """
        super().__init__()
        self.num_class = num_class
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.train_f1 = pl.metrics.F1(num_classes=self.num_class, average="macro")
        self.valid_f1 = pl.metrics.F1(num_classes=self.num_class, average="macro")
        self.test_f1 = pl.metrics.F1(num_classes=self.num_class, average="macro")

        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss
        self.params = params if params else {}

    def configure_optimizers(self) -> Any:
        """Configure optimizer for pytorch lighting.

        Returns: optimizer for pytorch lighting.

        """
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        """Get training step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        text, labels = batch
        output = self.forward(text)
        loss = self.loss(output, labels)
        self.log('train_loss', loss)
        self.train_acc(output, labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.train_f1(output, labels)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Get validation step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        text, labels = batch
        output = self.forward(text)
        loss = self.loss(output, labels)
        self.log('valid_loss', loss, on_epoch=True)
        self.valid_acc(output, labels)
        self.log('valid_acc', self.valid_acc, on_epoch=True)
        self.valid_f1(output, labels)
        self.log('valid_f1', self.valid_f1, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Get test step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        text, labels = batch
        output = self.forward(text)
        loss = self.loss(output, labels)
        self.log('test_loss', loss, on_epoch=True)
        self.test_acc(output, labels)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.test_f1(output, labels)
        self.log('test_f1', self.test_f1, on_epoch=True)
        return loss

    def setup_logs(self):
        """Log Optim and Scheduler."""
        optim_configs = self.configure_optimizers()
        self.params["loss"] = self.loss
        if isinstance(optim_configs, tuple):
            self.params["optim"], self.params["scheduler"] = optim_configs
            self.params["optim"] = self.params["optim"][0]
        else:
            self.params["optim"] = optim_configs

        for param in self.params["optim"].param_groups:
            param["params"] = None
