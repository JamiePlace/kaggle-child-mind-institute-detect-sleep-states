from typing import Any, Optional
import timeit
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


from src.conf import TrainConfig
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg
from src.models.prectimemodel import PrecTimeModel


class PrecTime(LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        feature_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg
        self.model = PrecTimeModel(
            cfg, in_channels=feature_dim, n_classes=num_classes
        )
        self.__best_loss = np.inf
        self.validation_loss = []
        self.val_loss_non_improvement = 0
        self.best_state_dict: dict = {}
        # TODO - implement weighted loss
        # this is done with one hot encoding the label
        # and assigning a weight to each class
        # see https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        self.training_step_outputs = {"preds": [], "labels": []}
        self.validation_step_outputs = {"preds": [], "labels": []}

    def forward(self, x: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, predictions = self.loss_caclulation(batch)
        self.log_dict(
            {
                "train_loss": loss.detach().item(),
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.training_step_outputs["preds"].append(predictions.argmax(dim=1))
        self.training_step_outputs["labels"].append(
            batch["sparse_label"].float()
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions = self.loss_caclulation(batch)
        self.log_dict(
            {
                "val_loss": loss.detach().item(),
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.validation_loss.append(loss.detach().item())
        self.validation_step_outputs["preds"].append(predictions.argmax(dim=1))
        self.validation_step_outputs["labels"].append(
            batch["sparse_label"].float()
        )
        # pr, re, f1 = self.calculate_metrics(
        #    batch["sparse_label"].cpu().numpy(),
        #    predictions.detach().cpu().numpy(),
        # )
        return loss

    def loss_caclulation(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(
            batch["feature"].float(),
        )
        sparse_label = batch["sparse_label"].float()
        # calculate class weights from sparse label
        # weights = self.calculate_pos_weights(sparse_label)
        predictions: torch.Tensor = output["predictions"]
        # convert sparse label to one hot encoding
        sparse_label = nn.functional.one_hot(
            sparse_label.long(), num_classes=2
        )
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(predictions, sparse_label.float())
        return loss, predictions

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.training_step_outputs["preds"])
        all_preds = all_preds.detach().cpu().numpy()
        all_labels = torch.cat(self.training_step_outputs["labels"])
        all_labels = all_labels.detach().cpu().numpy()

        all_preds = all_preds.round()
        pr, re, f1, ac, cm = self.calculate_metrics(
            labels=all_labels, preds=all_preds
        )
        cm = cm.rename({"labels": "Train"})
        print(f"Train: Precision: {pr}, Recall: {re}, F1: {f1}, Acc: {ac}")
        print(cm)
        self.training_step_outputs["preds"].clear()
        self.training_step_outputs["labels"].clear()

    def on_validation_epoch_end(self):
        best = False
        loss = np.array(self.validation_loss).mean()

        all_preds = torch.cat(self.validation_step_outputs["preds"])
        all_preds = all_preds.detach().cpu().numpy()
        all_labels = torch.cat(self.validation_step_outputs["labels"])
        all_labels = all_labels.detach().cpu().numpy()

        all_preds = all_preds.round()
        pr, re, f1, ac, cm = self.calculate_metrics(
            labels=all_labels, preds=all_preds
        )
        cm = cm.rename({"labels": "Validation"})
        print(
            f"Validation: Precision: {pr}, Recall: {re}, F1: {f1}, Acc: {ac}"
        )
        print(cm)

        if loss < self.__best_loss:
            best = True
            torch.save(self.model.state_dict(), "best_model.pth")
            self.__best_loss = loss
            self.val_loss_non_improvement = 0
        else:
            self.val_loss_non_improvement += 1
            best = False

        if (
            self.val_loss_non_improvement
            > self.cfg.trainer.early_stopping_patience
        ):
            self.model.load_state_dict(torch.load("best_model.pth"))
            self.val_loss_non_improvement = 0
        self.validation_step_outputs["preds"].clear()
        self.validation_step_outputs["labels"].clear()

        if best:
            print(f"Saved best model {self.__best_loss} -> {loss}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        # scheduler = get_polynomial_decay_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.cfg.scheduler.num_warmup_steps,
        #    num_training_steps=self.trainer.max_steps,
        #    power=self.cfg.scheduler.power,
        # )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.scheduler.num_warmup_steps,
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @staticmethod
    def calculate_metrics(
        labels: torch.Tensor, preds: torch.Tensor
    ) -> tuple[float, float, float, float, pl.DataFrame]:
        pr, re, f1, _ = precision_recall_fscore_support(
            labels, preds.round(), average="binary", zero_division=0
        )
        accuracy = (labels == preds).mean()
        cm = confusion_matrix(labels, preds)
        cm_df = pl.DataFrame(
            {
                "labels": ["label_0", "label_1"],
                "label_0": cm[:, 0],
                "label_1": cm[:, 1],
            }
        )
        return round(pr, 4), round(re, 4), round(f1, 4), round(accuracy, 2), cm_df  # type: ignore

    def calculate_pos_weights(self, labels: torch.Tensor) -> torch.Tensor:
        classes, class_counts = torch.unique(labels, return_counts=True)
        pos_weights = torch.ones_like(class_counts)
        neg_counts = [len(labels) - pos_count for pos_count in class_counts]
        for idx, (pos_count, neg_count) in enumerate(
            zip(class_counts, neg_counts)
        ):
            pos_weights[idx] = neg_count / (pos_count + 1e-5)

        return torch.as_tensor(pos_weights, dtype=torch.float)
