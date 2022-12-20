import os
from typing import *
import json
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from pytorch_lightning.loggers import WandbLogger



def predict_fn(pl_module, pl_data_module, split: str):
    
    result, labels = [], []
    assert split in ["train", "dev", "test", "top"], f'predict_fn() Error {split} not in ["train", "dev", "test", "top"]'

    dataset_selection: dict = {
        "train": pl_data_module.train_dataset, "dev": pl_data_module.dev_dataset, "test": pl_data_module.test_dataset
    }
    
    
    dataset = dataset_selection[split] if split in dataset_selection else pl_data_module.get_top_dataset()


    dataloader = DataLoader(dataset, shuffle=False, batch_size=2048)
    for batch in tqdm(dataloader, desc="predict"):
        logits = pl_module.model.forward(batch["features"].float(), dump=True)
        preds = logits.softmax(dim=-1).argmax(dim=-1).tolist()
        result.extend(preds)
        labels.extend(batch["labels"])
    return result, labels

def dump_predictions(predictions, labels, predict_path: str, split: str, is_top_dataset: bool = False):
    assert split in ["train", "dev", "test"] and len(predictions) == len(labels), "dump_predictions() error in arguments"
    top_postfix: str = '_top2' if is_top_dataset else ''
    with open(f"{predict_path}/predictions_{split}{top_postfix}.txt", "w") as pred_writer, open(f"{predict_path}/labels_{split}{top_postfix}.txt", "w") as label_writer:
        for pred, label in zip(predictions, labels):
            pred_writer.write(str(pred) + '\n')
            label_writer.write(str(int(label.item())) + '\n')


def dump_results(path, output_metrics_model: dict, split: str):
    assert split in ["train", "dev", "test"]
    path_output: str = "/".join(path.split("/")[:-1])
    print(path_output)
    with open(hydra.utils.to_absolute_path(f"{path_output}/result_{split}.json"), "w") as writer:
        json.dump(output_metrics_model[0], writer, indent=4, sort_keys=True)

def set_fast_dev_run(conf: DictConfig):
    if conf.train.pl_trainer.fast_dev_run:
        print(f"Debug mode <{conf.train.pl_trainer.fast_dev_run}>. Forcing debugger configuration")
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.gpus = 0
        conf.train.pl_trainer.precision = 32
        conf.data.num_workers = 0
        # Switch wandb mode to offline to prevent online logging
        conf.logging.wandb_arg.mode = "offline"


def gpus(conf: DictConfig) -> int:
    """Utility to determine the number of GPUs to use."""
    return conf.train.pl_trainer.gpus if torch.cuda.is_available() else 0


def enable_16precision(conf: DictConfig) -> int:
    """Utility to determine the number of GPUs to use."""
    return conf.train.pl_trainer.precision if torch.cuda.is_available() else 32


def set_determinism_the_old_way(deterministic: bool) -> None:
    # determinism for cudnn
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)


def get_number_of_cpu_cores() -> int:
    return os.cpu_count()


def build_callbacks(conf: DictConfig) -> Tuple[List[pl.Callback], Optional[ModelCheckpoint]]:
    """
    Add here your pytorch lightning callbacks
    """
    callbacks_store = [RichProgressBar()]
    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    model_checkpoint_callback: Optional[ModelCheckpoint] = None
    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)
    return callbacks_store, model_checkpoint_callback


def build_wandb_logger(conf):
    logger: Optional[WandbLogger] = None
    if conf.logging.log and not conf.train.pl_trainer.fast_dev_run:
        hydra.utils.log(f"Instantiating Wandb Logger")
        logger = hydra.utils.instantiate(conf.logging.wandb_arg)
    return logger
