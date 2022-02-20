from torch.utils.data import DataLoader

import os
import argparse
from pathlib import Path
import json

from training import *
from training import write_stats_to_tensorboard
from training_perceiver import *

from model import EventTransformer
from dsec import DSEC

from plot import save_plot_flow, create_event_picture

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



    save_figures_loss = False

    #vis_train_frames = list(zip(*([iter(args.vis_train_frames)] * 2)))
    #vis_val_frames = list(zip(*([iter(args.vis_val_frames)] * 2)))

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

    

    epochs = config['training']['epochs']
    lr = config['training']['lr']

    print_freq = 1
    val_freq = 2
    checkpoint_freq = config['output']['checkpoint_freq']


    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    train_set = DSEC(**config['train_set'])

    # TODO: mulitple validation sets
    val_sets = [DSEC(**config['val_sets'][i])
                for i in range(len(config['val_sets']))]


    print("Training set length: " + str(len(train_set)))
    for val_set in val_sets :
        print("Validation set length: " + str(len(val_set)))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               collate_fn=collate_dict_list,
                                               shuffle=True,
                                               pin_memory=True,
                                               **config['data_loader'])
    batch_size = train_loader.batch_size

    model = EventTransformer(**config['model'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                          start_factor=config['training'].get('warm_up_init', 1.),
                                                          total_iters=config['training'].get('warm_up_length', 0))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_halflife'], gamma=0.5)

    train_loss_history = []
    val_loss_histories = [[] for _ in val_sets]
    val_loss_steps = []
    epoch_0 = 0

    if args.checkpoint is not None :
        checkpoint = torch.load(args.checkpoint)
        epoch_0 = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        train_loss_history = checkpoint['train_loss_history']
        val_loss_histories = checkpoint['val_loss_histories']
        val_loss_steps = checkpoint['val_loss_steps']
        random.setstate(checkpoint['py_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])

    LFunc = torch.nn.L1Loss()
    print(type(LFunc))


    writer = tensorboard.SummaryWriter(log_dir=output_path, purge_step=epoch_0)

    for it in range(epoch_0, epochs) :
        epoch_begin = time.time()
        if it == args.finetune_epoch :
            train_set.set_crop()
            if args.finetune_lr :
                optimizer = torch.optim.Adam(model.parameters(), args.finetune_lr)
                warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                                      start_factor=config['training'].get(
                                                                          'warm_up_init', 1.),
                                                                      total_iters=config['training'].get(
                                                                          'warm_up_length', 0))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_halflife, gamma=0.5)
                warm_up_scheduler.step(it-1)
                scheduler.step(it-1)
            if args.finetune_batch_size :
                train_loader = torch.utils.data.DataLoader(train_set,
                                                           batch_size=args.finetune_batch_size,
                                                           collate_fn=collate_dict_list,
                                                           num_workers=args.num_workers, shuffle=True,
                                                           pin_memory=True)
            train_set.random_backward = False

        report = process_epoch(it, model, LFunc, train_set, device,
                               forward_perceiver, eval_perceiver_out,
                               train_loader, optimizer,
                               config['output']['vis_train_frames'] if it % config['output']['vis_freq'] == 0 else [],
                               batch_size=batch_size if not train_set.batch_backward else batch_size * 2,
                               full_query=config['training'].get('full_query', False),
                               predict_targets=config['training'].get('predict_targets', False))
        # print(report['runtime'])
        warm_up_scheduler.step()
        scheduler.step()

        write_stats_to_tensorboard(report['stats'], writer, it, prefix="train")
        train_loss_history.append(report['stats']['loss'])

        # im_grid = torchvision.utils.make_grid([torch.tensor(im.transpose((2, 0,1 )))
        #                                        for im in stats['ims'].values()])
        for im_name in report['ims'].keys() :
            writer.add_image("train/" + im_name,
                             report['ims'][im_name].transpose((2, 0, 1)), it)

        if it % print_freq == 0 :
            it_string = ("Iteration: " + str(it) + ", training loss : " +
                  "{0:.3f}".format(report['stats']['loss']) +
                  ", time: " + "{0:.1f}".format(time.time() - epoch_begin))
            print(it_string)
            with open(output_path / "output.txt", 'w') as file:
                file.write(it_string)


        if it % val_freq == 0 or it % config['output']['vis_freq'] == 0 :
            model.eval()

            for idx_val_set, val_set in enumerate(val_sets) :
                with torch.no_grad() :
                    report = process_epoch(it, model, LFunc, val_set, device,
                                           forward_perceiver, eval_perceiver_out,
                                           vis_frames=config['output']['vis_val_frames']
                                           if it % config['output']['vis_freq'] == 0 else [],
                                           full_query=config['training'].get('full_query', False),
                                           predict_targets=config['training'].get('predict_targets', False))

                    # TODO: print frame
                print("Iteration: " + str(it) + ", validation loss : " +
                      "{0:.3f}".format(report['stats']['loss']))
                val_loss_histories[idx_val_set].append(report['stats']['loss'])

                write_stats_to_tensorboard(report['stats'], writer, it, prefix="val" + str(idx_val_set))

                for im_name in report['ims'].keys():
                    writer.add_image("val" + str(idx_val_set) + "/" + im_name,
                                     report['ims'][im_name].transpose((2, 0, 1)), it)

            val_loss_steps.append(it)

            if save_figures_loss :
                plt.plot(np.arange(0, len(train_loss_history)), train_loss_history, label="train")
                for i, val_loss_history in enumerate(val_loss_histories) :
                    if len(val_loss_history) == len(val_loss_steps):
                        plt.plot(val_loss_steps, val_loss_history, label="val"+str(i))
                plt.legend(loc='upper right')
                plt.savefig(output_path / "loss_history.png",
                            pad_inches=0.)
                plt.close()

            model.train()

        if it % checkpoint_freq == 0 :
            checkpoint = {
                'epoch' : it + 1,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'warmup' : warm_up_scheduler.state_dict(),
                'train_loss_history' : train_loss_history,
                'val_loss_histories' : val_loss_histories,
                'val_loss_steps' : val_loss_steps,
                'py_rng' : random.getstate(),
                'torch_rng' : torch.get_rng_state()
            }
            torch.save(checkpoint, output_path / ("checkpoint" + str(it)))

    print(torch.cuda.max_memory_allocated(device=device))

    writer.flush()
    writer.close()

if __name__ == "__main__" :
    main()
