# models/model.py
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryF1Score


# ImageClassifier для инференса
class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name='vit_base_patch16_clip_224.openai'):
        super().__init__()
        self.backbone = timm.create_model(model_name=model_name, pretrained=True, num_classes=1,
                                         drop_rate=0.3, attn_drop_rate=0.1, drop_path_rate=0.1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # Метрики
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()

    def forward(self, x):
        # x: (B, C, H, W) для ImageClassifier
        logits = self.backbone(x) # (B, 1)
        return logits.squeeze(-1) # (B,) или (B,) в зависимости от размера батча

    def training_step(self, batch, batch_idx):
        x, y = batch[:2] # x: (B, C, H, W), y: (B,)
        logits = self(x) # (B,)
        loss = self.loss_fn(logits, y) # (B,) vs (B,)

        probs = torch.sigmoid(logits)
        self.train_auroc.update(probs, y)
        self.train_f1.update(probs, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_auroc', self.train_auroc.compute(), prog_bar=True)
        self.log('train_f1', self.train_f1.compute(), prog_bar=True)
        self.train_auroc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x) # (B,)
        loss = self.loss_fn(logits, y) # (B,) vs (B,)

        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, y)
        self.val_f1.update(probs, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), prog_bar=True)
        self.val_auroc.reset()
        self.val_f1.reset()