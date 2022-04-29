import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.data
import torch.nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
from pathlib import Path
import json

from training.training import *
from training.training_perceiver import *
from training.training_perceiver_imflow import *
from training.training_unet import EventUnetTrainer
from training.training_eraft import ERaftTrainer

from model import EventTransformer
from UNet.model import UNet
from ERAFT.model.eraft import ERAFT

from data.event_datasets import EventFlowDataset, LimitedEventsDataset

from data.utils import collate_dict_list


from train import set_up, update
from training.training_lightning import TrainingModule
"""
torch.manual_seed(13)
import random
random.seed(13)
np.random.seed(13)
"""
pl.seed_everything(7)


parser = argparse.ArgumentParser()

parser.add_argument("--output_path", type=str)
parser.add_argument("--setting", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--config_file", type=open, nargs='+')
parser.add_argument("--resume", action='store_true')
parser.add_argument("--checkpoint_load_model_only", action='store_true')

def main() :
    args = parser.parse_args()
    config = {}

    if args.config_file is not None:
        for cfg in args.config_file:
            config = update(config, json.load(cfg))

    if hasattr(args, 'output_path') and args.output_path is not None:
        save_figures_loss = True
        output_path = Path(args.output_path)
        print(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            with open(output_path / "settings.txt", 'w') as file:
                file.write("\n".join("{}: {}".format(k, v) for k, v in vars(args).items()))

            args.resume = False

        if not args.resume:
            with open(output_path / "config.json", 'w') as file:
                file.write(json.dumps(config, indent=4))
        if args.resume:
            existing_files = sorted(os.listdir(output_path))
            if "config.json" in existing_files:
                print("Resuming config: " + str(output_path / "config.json"))
                config = json.load(open(output_path / "config.json"))
            checkpoints = [f for f in existing_files if "checkpoint" in f]
            latest = np.argmax([int(f.split("checkpoint")[1]) for f in checkpoints])
            print("Resuming checkpoint: " + str(output_path / checkpoints[latest]))
            args.checkpoint = output_path / checkpoints[latest]
            # TODO: how to find latest config file

    print(json.dumps(config, indent=4))
    #config_add_defaults(config)

    epochs = config['training']['epochs']

    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    epoch_0 = 0
    config.setdefault('dataset', {})

    model, train_set, val_sets, model_trainer, config = set_up(config)

    if args.checkpoint is not None :
        model.load_state_dict(torch.load(args.checkpoint)['model'])

    print(len(train_set))
    print(len(val_sets[0]))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               collate_fn=train_set.collate,
                                               **config['data_loader'])
    val_loader = torch.utils.data.DataLoader(val_sets[0], collate_fn=val_sets[0].collate,
                                             **{**config['data_loader'], 'batch_size' : 1})

    print(len(train_loader))

    pl_training_module = TrainingModule(model, model_trainer)


    tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_path)
    tb_logger = CustomLogger(save_dir=output_path)
    


    progress_bar_callback = pl.callbacks.progress.TQDMProgressBar(refresh_rate=200)

    trainer = pl.Trainer(max_epochs = epochs, logger=tb_logger, **config['training']['lightning'],
                         log_every_n_steps=1,
                         enable_progress_bar=False,
                         check_val_every_n_epoch=2, # TODO: read from config file
                         val_check_interval=None)

    trainer.fit(pl_training_module, train_loader)


class CustomLogger(pl_loggers.TensorBoardLogger) :
    def log_metrics(self, metrics, step) :
        return super().log_metrics(metrics, step)


if __name__ == "__main__" :
    main()
