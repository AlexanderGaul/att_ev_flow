import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import numpy as np

from training.training import TrainerTraining
from training.training_perceiver_imflow import ImagePerceiverTrainer
from data.image_data import ImageFlowDataset
from model import EventTransformer

from pathlib import Path
import json

torch.manual_seed(2)
import random
random.seed(2)
np.random.seed(2)

parser = argparse.ArgumentParser()

parser.add_argument("--output_path", type=str)
parser.add_argument("--message")
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--config_file", type=open)
parser.add_argument("--resume", action='store_true')


def main() :
    args = parser.parse_args()
    config = {"training" : {},
              "model" : {},
              "train_set" : {},
              "val_set" : {},
              "data_loader" : {},
              "output" : {}}

    if args.config_file is not None :
        config = json.load(args.config_file)

    if hasattr(args, 'output_path') and args.output_path is not None :
        save_figures_loss = True
        output_path = Path(args.output_path)
        print(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            with open(output_path / "settings.txt", 'w') as file :
                file.write("\n".join("{}: {}".format(k, v) for k, v in vars(args).items()))

            with open(output_path / "config.json", 'w') as file :
                file.write(json.dumps(config, indent=4))

            args.resume = False

        elif args.resume :
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

    epochs = config['training']['epochs']

    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model = EventTransformer(**config['model'],
                             input_format={'xy':[0, 1],
                                           't' : [],
                                           'raw' : range(2, 56)})
    model.to(device)

    model_trainer = ImagePerceiverTrainer(lfunc=torch.nn.L1Loss())

    epoch_0 = 0

    train_set = ImageFlowDataset(**config['train_set'])

    val_sets = [ImageFlowDataset(**config['val_sets'][i])
                     for i in range(len(config['val_sets']))]

    training = TrainerTraining(model, train_set, val_sets,
                               model_trainer, config, device, output_path,
                               train_set.collate)

    if args.checkpoint :
        print("Loading checkpoint: " + str(args.checkpoint))
        training.load_checkpoint(args.checkpoint)

    for it in range(epoch_0, epochs) :
        training.run_epoch()

    print(torch.cuda.max_memory_allocated(device=device))

if __name__ == "__main__" :
    main()