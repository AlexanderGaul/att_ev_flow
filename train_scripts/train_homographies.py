import os
import argparse
from pathlib import Path
import json

from training.training import *
from training.training_perceiver import *

from model import EventTransformer
from edata import HomographyDataset

from utils import collate_dict_list

torch.manual_seed(2)
import random
random.seed(2)
np.random.seed(2)

matplotlib.use('Agg')

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


# Training options
training_args = parser.add_argument_group("Training")
#training_args.add_argument("--lr", type=float, default=0.0003)
"""training_args.add_argument("--lr_halflife", type=float, default=100000000)
training_args.add_argument("--batch_size", type=int, default=1)
training_args.add_argument("--epochs", type=int, default=50)
training_args.add_argument("--full_query", action='store_true')
training_args.add_argument("--predict_targets", action='store_true')"""

#training_args.add_argument("--warm_up_init", type=float, default=1.)
#training_args.add_argument("--warm_up_length", type=int, default=0)
training_args.add_argument("--finetune_epoch", type=int, default=-1)
training_args.add_argument("--finetune_lr", type=float, default=None)
training_args.add_argument("--finetune_batch_size", type=int, default=None)



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


    model = EventTransformer(**config['model'])
    model.to(device)

    epoch_0 = 0

    LFunc = torch.nn.L1Loss()

    training = Training(model, HomographyDataset,
                        forward_perceiver, eval_perceiver_out, LFunc,
                        config, device,
                        output_path,
                        collate_dict_list)

    if args.checkpoint :
        print("Loading checkpoint: " + str(args.checkpoint))
        training.load_checkpoint(args.checkpoint)

    for it in range(epoch_0, epochs) :
        training.step()

    print(torch.cuda.max_memory_allocated(device=device))

if __name__ == "__main__" :
    main()
