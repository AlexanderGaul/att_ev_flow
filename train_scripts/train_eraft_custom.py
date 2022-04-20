import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from pathlib import Path
import json

from ERAFT.model.eraft import ERAFT

from training.training_eraft import *

from dsec import DSEC

from data.utils import collate_dict_list

torch.manual_seed(2)
import random
random.seed(2)
np.random.seed(2)

# matplotlib.use('Agg')

parser = argparse.ArgumentParser()

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)

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

    if hasattr(args, 'output_path') and args.output_path is not None:
        save_figures_loss = True
        output_path = Path(args.output_path)
        print(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            with open(output_path / "settings.txt", 'w') as file:
                file.write("\n".join("{}: {}".format(k, v) for k, v in vars(args).items()))

            with open(output_path / "config.json", 'w') as file:
                file.write(json.dumps(config, indent=4))

            args.resume = False

        elif args.resume:
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

    epochs = config['training']['epochs']

    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    config_eraft = {'name': 'dsec',
              'cuda': True,
              'gpu': 0,
              'subtype': 'standard',
              'save_dir': 'saved',
              'data_loader': {
                  'train': {
                      'args': {'batch_size': config['data_loader']['batch_size'],
                               'shuffle': False,
                               'sequence_length': 1,
                               'num_voxel_bins': config['train_set']['num_bins']}}}}

    model = ERAFT(config_eraft, config_eraft['data_loader']['train']['args']['num_voxel_bins'])
    model.to(device)

    epoch_0 = 0

    LFunc = torch.nn.L1Loss()

    training = Training(model, DSEC,
                        forward_eraft_custom, eval_eraft_custom, LFunc,
                        config, device,
                        output_path,
                        collate_fn=collate_dict_list)

    if args.checkpoint :
        print("Loading checkpoint: " + str(args.checkpoint))
        training.load_checkpoint(args.checkpoint)

    for it in range(epoch_0, epochs) :
        training.step()

    print(torch.cuda.max_memory_allocated(device=device))

if __name__ == "__main__" :
    main()
