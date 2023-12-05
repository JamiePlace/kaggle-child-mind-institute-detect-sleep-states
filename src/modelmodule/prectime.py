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
from src.models.prectimemodel import PrecTimeModel
from run.inference import make_predictions, make_submission
from run.examine_results import calculate_score


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
        self.training_step_outputs = {
            "keys": [],
            "sparse_preds": [],
            "sparse_labels": [],
            "dense_preds": [],
            "dense_labels": [],
        }
        self.validation_step_outputs = {
            "keys": [],
            "sparse_preds": [],
            "sparse_labels": [],
            "dense_preds": [],
            "dense_labels": [],
        }

    def forward(self, x: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, sparse_preds, dense_preds = self.loss_caclulation(batch)
        self.log_dict(
            {
                "train_loss": loss.detach().item(),
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.training_step_outputs["keys"].append(batch["key"])
        self.training_step_outputs["sparse_preds"].append(sparse_preds)
        self.training_step_outputs["dense_preds"].append(dense_preds)
        self.training_step_outputs["sparse_labels"].append(
            batch["sparse_label"].float()
        )
        self.training_step_outputs["dense_labels"].append(
            batch["dense_label"].float()
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, sparse_preds, dense_preds = self.loss_caclulation(batch)
        self.log_dict(
            {
                "val_loss": loss.detach().item(),
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.validation_step_outputs["keys"].append(batch["key"])
        self.validation_step_outputs["sparse_preds"].append(sparse_preds)
        self.validation_step_outputs["dense_preds"].append(dense_preds)
        self.validation_step_outputs["sparse_labels"].append(
            batch["sparse_label"].float()
        )
        self.validation_step_outputs["dense_labels"].append(
            batch["dense_label"].float()
        )
        self.validation_loss.append(loss.detach().item())
        return loss

    def loss_caclulation(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch["feature"].shape) == 4:
            batch["feature"] = batch["feature"].squeeze()
            batch["dense_label"] = batch["dense_label"].squeeze()
            batch["sparse_label"] = batch["sparse_label"].squeeze()
        output = self.model(
            batch["feature"].float(),
        )
        dense_predictions: torch.Tensor = output["dense_predictions"]
        dense_label = batch["dense_label"].float()
        loss_dense = self.dense_loss_calculation(
            dense_label, dense_predictions
        )
        sparse_predictions: torch.Tensor = output["sparse_predictions"]
        sparse_label = batch["sparse_label"].float()
        loss_sparse = self.sparse_loss_calculation(
            sparse_label, sparse_predictions
        )

        return (
            loss_sparse + (self.cfg.dense_weight * loss_dense),
            sparse_predictions,
            dense_predictions,
        )

    def sparse_loss_calculation(
        self, sparse_label: torch.Tensor, sparse_predictions: torch.Tensor
    ) -> torch.Tensor:
        # convert sparse label to one hot encoding
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(sparse_predictions, sparse_label.float())
        return loss

    def dense_loss_calculation(
        self, dense_label: torch.Tensor, dense_predictions: torch.Tensor
    ) -> torch.Tensor:
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(dense_predictions, dense_label.float())

    def on_train_epoch_end(self):
        print(max(self.training_step_outputs["dense_preds"]))
        self.training_step_outputs["keys"].clear()
        self.training_step_outputs["dense_preds"].clear()
        self.training_step_outputs["sparse_preds"].clear()
        self.training_step_outputs["dense_labels"].clear()
        self.training_step_outputs["sparse_labels"].clear()

    def on_validation_epoch_end(self):
        best = False
        loss = np.array(self.validation_loss).mean()

        all_preds = torch.cat(self.validation_step_outputs["dense_preds"])
        all_preds = all_preds.view(all_preds.shape[0] * all_preds.shape[1], -1)
        all_preds = all_preds.detach().cpu().numpy()
        all_labels = torch.cat(self.validation_step_outputs["dense_labels"])
        all_labels = all_labels.view(
            all_labels.shape[0] * all_labels.shape[1], -1
        )
        all_labels = all_labels.detach().cpu().numpy()

        structured_preds = make_predictions(
            self.validation_step_outputs["keys"],
            self.validation_step_outputs["dense_preds"],
        )
        submission_df = make_submission(self.cfg, structured_preds)
        score = calculate_score(self.cfg, pred_df=submission_df)

        self.log_dict(
            {
                "val_score": score,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

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
        self.validation_step_outputs["keys"].clear()
        self.validation_step_outputs["dense_preds"].clear()
        self.validation_step_outputs["sparse_preds"].clear()
        self.validation_step_outputs["dense_labels"].clear()
        self.validation_step_outputs["sparse_labels"].clear()
        self.validation_loss.clear()

        if best:
            print(f"Saved best model {self.__best_loss} -> {loss}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        # scheduler = get_polynomial_decay_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.cfg.scheduler.num_warmup_steps,
        #    num_training_steps=self.trainer.max_steps,
        #    power=self.cfg.scheduler.power,
        #    lr_end=0,
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
            labels, preds.round(), average="weighted", zero_division=0
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
