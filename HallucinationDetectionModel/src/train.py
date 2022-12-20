#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

from typing import *
import json


import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.lightning_data_modules import LightningDataModule
from src.lightning_module import PytorchLightningDataModule
from src.utils.utilities import set_determinism_the_old_way, set_fast_dev_run, build_callbacks, build_wandb_logger, dump_results, dump_predictions, predict_fn


def train(conf: omegaconf.DictConfig) -> None:

    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(True)

    set_fast_dev_run(conf)

    pl_data_module: LightningDataModule = LightningDataModule(conf)
    pl_data_module.prepare_data()

    pl_module: PytorchLightningDataModule = PytorchLightningDataModule(conf)

    callbacks_store, model_checkpoint_callback = build_callbacks(conf)

    logger: Optional[WandbLogger] = build_wandb_logger(conf)

    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=logger)
    
    trainer.fit(pl_module, datamodule=pl_data_module)

    if conf.train.model_checkpoint_callback is not None and not trainer.fast_dev_run:
        pl_module = pl_module.load_from_checkpoint(checkpoint_path=model_checkpoint_callback.best_model_path)
        print("MODEL LOADED ", model_checkpoint_callback.best_model_path)

    pl_module.init_metrics()

    test_output = trainer.test(pl_module, datamodule=pl_data_module)
    dump_results(model_checkpoint_callback.best_model_path, test_output, "test")
    pl_module.init_metrics()

    val_output = trainer.test(pl_module, dataloaders=pl_data_module.val_dataloader())
    dump_results(model_checkpoint_callback.best_model_path, val_output, "dev")
    pl_module.init_metrics()


    train_output = trainer.test(pl_module, dataloaders=pl_data_module.train_dataloader())
    dump_results(model_checkpoint_callback.best_model_path, train_output, "train")
    pl_module.init_metrics()


    predict_path = '/'.join(model_checkpoint_callback.best_model_path.split("/")[:-1])

    #predictions, labels = pl_module.predict_fn(conf, pl_data_module, is_top_dataset=False)
    predictions, labels = predict_fn(pl_module, pl_data_module, "test")
    dump_predictions(predictions, labels, predict_path, "test", is_top_dataset=False)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
