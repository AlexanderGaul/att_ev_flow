import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn
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
from model_point_transformer import PointTransformer
from UNet.model import UNet
from ERAFT.model.eraft import ERAFT

from data.event_datasets import EventFlowDataset, LimitedEventsDataset

from data.utils import collate_dict_list

torch.manual_seed(2)
import random
random.seed(2)
np.random.seed(2)


parser = argparse.ArgumentParser()

parser.add_argument("--output_path", type=str)
parser.add_argument("--setting", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--config_file", type=open, nargs='+')
parser.add_argument("--resume", action='store_true')
parser.add_argument("--checkpoint_load_model_only", action='store_true')


import collections.abc


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def set_up(config) :
    DatasetClass = EventFlowDataset
    train_set = DatasetClass(**{**config['dataset'], **config['train_set']})
    val_sets = [DatasetClass(**{**config['dataset'], **config['val_sets'][i]})
                for i in range(len(config['val_sets']))]

    config['model']['input_format'] = train_set.data_format()

    if config['setup']['model'] == "EventTransformer" :
        model = EventTransformer(**config['model'])
    elif config['setup']['model'] == 'UNet' :
        model = UNet(train_set.data_format()['t_bins'], 2)
    elif config['setup']['model'] == 'ERaft' :
        model = ERAFT({'subtype' : 'standard'}, n_first_channels=train_set.data_format['t_bins'], **config['model'])

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    print(get_n_params(model))

    if config['setup']['model'] == "EventTransformer" :
        if train_set.event_images:
            model_trainer = ImagePerceiverTrainer(lfunc=torch.nn.L1Loss(), **config['training']['params'])
        else:
            model_trainer = EventPerceiverTrainer(lfunc=torch.nn.L1Loss(), **config['training']['params'])
    elif config['setup']['model'] == 'UNet' :
        model_trainer = EventUnetTrainer(lfunc=torch.nn.L1Loss(), **config['training']['params'])
    elif config['setup']['model'] == 'ERaft' :
        model_trainer = ERaftTrainer(lfunc=torch.nn.L1Loss())

    return model, train_set, val_sets, model_trainer, config



def main() :
    args = parser.parse_args()
    config = {}

    if args.config_file is not None :
        for cfg in args.config_file :
            config = update(config, json.load(cfg))

    if hasattr(args, 'output_path') and args.output_path is not None :
        save_figures_loss = True
        output_path = Path(args.output_path)
        print(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            with open(output_path / "settings.txt", 'w') as file :
                file.write("\n".join("{}: {}".format(k, v) for k, v in vars(args).items()))

            args.resume = False

        if not args.resume :
            with open(output_path / "config.json", 'w') as file:
                file.write(json.dumps(config, indent=4))
        if args.resume :
            existing_files = sorted(os.listdir(output_path))
            if "config.json" in existing_files :
                print("Resuming config: " + str(output_path / "config.json"))
                config = json.load(open(output_path / "config.json"))
            checkpoints = [f for f in existing_files if "checkpoint" in f]
            latest = np.argmax([int(f.split("checkpoint")[1]) for f in checkpoints])
            print("Resuming checkpoint: " + str(output_path / checkpoints[latest]))
            args.checkpoint = output_path / checkpoints[latest]
            # TODO: how to find latest config file


    print(json.dumps(config, indent=4))
    config_add_defaults(config)

    epochs = config['training']['epochs']

    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    epoch_0 = 0
    config.setdefault('dataset', {})


    model, train_set, val_sets, model_trainer, config = set_up(config)
    model.to(device)

    training = TrainerTraining(model, train_set, val_sets, model_trainer,
                               config, device,
                               output_path,
                               train_set.collate)

    if args.checkpoint :
        print("Loading checkpoint: " + str(args.checkpoint))
        training.load_checkpoint(args.checkpoint, args.checkpoint_load_model_only)

    for it in range(epoch_0, epochs) :
        training.run_epoch()

    print(torch.cuda.max_memory_allocated(device=device))

if __name__ == "__main__" :
    main()