import logging
from pathlib import Path
import os

import hydra
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from src.conf import TrainConfig
from src.datamodule.seg import SegDataModule, pre_process_for_training
from src.modelmodule.seg import SegModel

# performance due to my class gpu
torch.set_float32_matmul_precision("medium")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
)
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    # pre-process data to enable efficient data loading

    train_data_files = [
        train_file.name
        for train_file in (Path(cfg.dir.processed_dir) / "train").glob("*.pkl")
    ]
    if len(train_data_files) > 0:
        LOGGER.info("Removing Previously Processed Files")
        for file in train_data_files:
            os.remove(Path(cfg.dir.processed_dir) / "train" / file)

    LOGGER.info("Processing Data for Loading")
    pre_process_for_training(cfg)

    seed_everything(cfg.seed)
    # init lightning model
    datamodule = SegDataModule(cfg)
    LOGGER.info("Set Up DataModule")
    model = SegModel(
        cfg,
        datamodule.valid_event_df,
        len(cfg.features),
        len(cfg.labels),
        cfg.duration,
    )

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.trainer.monitor,
        mode=cfg.trainer.monitor_mode,
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project="child-mind-institute-detect-sleep-states",
    )

    pl_logger.log_hyperparams(cfg)  # type: ignore

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.trainer.accelerator,
        precision="16-mixed" if cfg.trainer.use_amp else "32",
        # training
        fast_dev_run=cfg.trainer.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.trainer.epochs,
        max_steps=cfg.trainer.epochs * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    # this doesn't work smh
    # TODO fix
    trainer.fit(model, datamodule=datamodule)

    # load best weights
    model = model.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )
    weights_path = str("model_weights.pth")  # type: ignore
    LOGGER.info(f"Extracting and saving best weights: {weights_path}")
    torch.save(model.model.state_dict(), weights_path)

    return


if __name__ == "__main__":
    main()
