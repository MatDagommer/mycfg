"""CNN model definition for Fashion-MNIST classification."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionMNISTCNN(pl.LightningModule):
    """
    Convolutional Neural Network for Fashion-MNIST classification.

    Args:
        num_classes (int): Number of output classes. Default is 10.

    Returns:
        nn.Module: CNN model instance.
    """
    def __init__(self, num_classes=10):
        """Initialize the CNN model."""
        super(FashionMNISTCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass of the CNN model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for PyTorch Lightning."""
        x, y = batch if isinstance(batch, (list, tuple)) \
            and len(batch) == 2 else (batch, None)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """Load model from checkpoint with pre-trained weights."""
        return super().load_from_checkpoint(checkpoint_path, **kwargs)
