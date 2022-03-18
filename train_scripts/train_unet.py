import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import json

from training.training import *
from training.training_unet import *

from UNet.model import UNet
from dsec import DSEC

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


"""parser.add_argument("--train_seqs", nargs='+', type=int, default=list(range(1, 18)))
parser.add_argument("--val_seqs", nargs='+', action='append', type=int, default=[])
parser.add_argument("--num_workers", type=int, default=8)
"""
# Dataset options
"""parser.add_argument("--dt", type=int, default=0)
parser.add_argument("--include_backward", action='store_true')
parser.add_argument("--random_backward", action='store_true')
parser.add_argument("--batch_backward", action='store_true')
parser.add_argument("--add_previous_frame", action='store_true')
parser.add_argument("--crop", nargs=2, type=int)
parser.add_argument("--random_crop_offset", action='store_true')
parser.add_argument("--fixed_crop_offset", nargs=2, type=int)
parser.add_argument("--val_fixed_crop_offset", nargs=2, type=int)
parser.add_argument("--random_flip_horizontal", action='store_true')
parser.add_argument("--random_flip_vertical", action='store_true')
parser.add_argument("--random_moving", action='store_true')
parser.add_argument("--random_dt", action='store_true')
parser.add_argument("--scale_crop", action='store_true')
parser.add_argument("--crop_keep_full_res", action='store_true')
parser.add_argument("--sum_groups", type=int, default=0)
parser.add_argument("--event_set", type=str, default='left')"""

# Model settings
"""model_args = parser.add_argument_group("Model")
model_args.add_argument("--xy_encoding_bands", type=int, default=16)
model_args.add_argument("--t_encoding_bands", type=int, default=16)
model_args.add_argument("--depth", type=int, default=8)
model_args.add_argument("--perceiver_params", nargs='+', default=[])"""

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

# Output options
"""parser.add_argument("--vis_freq", type=int, default=20)
parser.add_argument("--vis_train_frames", nargs='+', type=int, default=[])
parser.add_argument("--vis_val_frames", nargs='+', type=int, default=[])
parser.add_argument("--checkpoint_freq", type=int, default=10)"""




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

    print(json.dumps(config, indent=4))

    assert config['train_set']['build_volume']

    epochs = config['training']['epochs']

    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model = UNet(config['train_set']['num_bins'], 2)
    model.to(device)

    epoch_0 = 0

    LFunc = torch.nn.L1Loss()

    training = Training(model, DSEC,
                        forward_unet, eval_unet, LFunc,
                        config, device,
                        output_path,
                        collate_dict_list)

    for it in range(epoch_0, epochs) :
        training.step()

    print(torch.cuda.max_memory_allocated(device=device))

if __name__ == "__main__" :
    main()
