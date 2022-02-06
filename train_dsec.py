import math

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
import torchvision

import matplotlib
import matplotlib.pyplot as plt

import os
import argparse
from pathlib import Path

import time

from model import EventTransformer
from dsec import DSEC, get_grid_coordinates

from plot import get_np_plot_flow, save_plot_flow, create_event_picture, plot_flow_color, plot_flow_error

from utils import collate_dict_list, default

torch.manual_seed(1)
import random
random.seed(1)
np.random.seed(1)

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
parser.add_argument("--args_file", type=open, action=LoadFromFile)

parser.add_argument("--train_seqs", nargs='+', type=int, default=list(range(1, 18)))
parser.add_argument("--val_seqs", nargs='+', action='append', type=int, default=[])
parser.add_argument("--num_workers", type=int, default=8)

# Dataset options
parser.add_argument("--dt", type=int, default=0)
parser.add_argument("--include_backward", action='store_true')
parser.add_argument("--crop", nargs=2, type=int)
parser.add_argument("--random_crop_offset", action='store_true')
parser.add_argument("--fixed_crop_offset", nargs=2, type=int)
parser.add_argument("--val_fixed_crop_offset", nargs=2, type=int)
parser.add_argument("--random_flip_horizontal", action='store_true')
parser.add_argument("--random_flip_vertical", action='store_true')
parser.add_argument("--random_moving", action='store_true')
parser.add_argument("--scale_crop", action='store_true')
parser.add_argument("--crop_keep_full_res", action='store_true')
parser.add_argument("--sum_groups", type=int, default=0)
parser.add_argument("--event_set", type=str, default='left')

# Model settings
model_args = parser.add_argument_group("Model")
model_args.add_argument("--xy_encoding_bands", type=int, default=16)
model_args.add_argument("--t_encoding_bands", type=int, default=16)
model_args.add_argument("--depth", type=int, default=8)
model_args.add_argument("--perceiver_params", nargs='+', default=[])

# Training options
training_args = parser.add_argument_group("Training")
training_args.add_argument("--lr", type=float, default=0.0003)
training_args.add_argument("--lr_halflife", type=float, default=100000000)
training_args.add_argument("--batch_size", type=int, default=1)
training_args.add_argument("--epochs", type=int, default=50)
training_args.add_argument("--full_query", action='store_true')
training_args.add_argument("--predict_targets", action='store_true')
training_args.add_argument("--warm_up_init", type=float, default=1.)
training_args.add_argument("--warm_up_length", type=int, default=0)
training_args.add_argument("--finetune_epoch", type=int, default=-1)

# Output options
parser.add_argument("--vis_freq", type=int, default=20)
parser.add_argument("--vis_train_frames", nargs='+', type=int, default=[])
parser.add_argument("--vis_val_frames", nargs='+', type=int, default=[])
parser.add_argument("--checkpoint_freq", type=int, default=10)



def save_plot_batch(path, event_data, locs, flows, frame_i, res):
    for i in range(len(event_data)):
        save_plot_flow(path + str(frame_i[i]) + ".png",
                       create_event_picture(event_data[i], res=res),
                       locs[i],
                       flows[i],
                       flow_res=res)

def write_stats_to_tensorboard(stats:dict, writer:tensorboard.SummaryWriter,
                               it:int,
                               prefix:str="", suffix:str="") :
    for k in stats.keys() :
        if type(stats[k]) is dict :
            write_stats_to_tensorboard(stats[k], writer, it,
                                       prefix=prefix+"/"+k, suffix=suffix)
        else :
            writer.add_scalar(prefix+"/"+k+"/"+suffix, stats[k], it)


def forward_perceiver(model, sample, device, dataset, full_query, predict_targets) :
    event_data = sample['in']['events']
    query_locs = sample['out']['coords']

    events_torch = [torch.tensor(e, dtype=torch.float32, device=device)
                    for e in event_data]

    if full_query:
        query_torch = torch.stack(len(event_data) *
                                  [torch.tensor(
                                      get_grid_coordinates(
                                          dataset.res, (0, 0)),
                                      dtype=torch.float32, device=device)])
    else:
        query_torch = [torch.tensor(l, dtype=torch.float32, device=device)
                       for l in query_locs]

    pred = model(events_torch, query_torch,
                 sample['in']['res'], sample['in']['dt'], sample['in']['tbins'])

    if predict_targets:
        for i in range(len(pred)):
            pred[i] = pred[i] - query_torch[i]

    return pred, query_torch


def eval_perceiver_out(out,
                       sample, device, lfunc, dense) :
    pred, query_torch = out

    if dense:
        pred = [predi[sample['out']['coords_grid_idx'][i], :]
                for i, predi in enumerate(pred)]

    flows = sample['out']['flows']

    flows_torch = [torch.tensor(flow_i, dtype=torch.float32, device=device)
                   for flow_i in flows]

    loss = torch.cat([lfunc(pred[i],
                            flows_torch[i]).reshape(1)
                      for i in range(len(pred))]).sum()

    return (loss,
            [predi.detach() for predi in pred],
            [f.detach() for f in flows_torch],
            sample['out']['coords'])


def compute_statistics(pred, flow, report, fraction) :
    for i in range(len(pred)):
        pred_norm = pred[i].norm(dim=1)
        flow_norm = flow[i].norm(dim=1)
        report['stats']['flow_length']['pred'] += \
            pred_norm.mean().item() / fraction
        report['stats']['flow_length']['gt'] += \
            flow_norm.mean().item() / fraction
        report['stats']['error_metrics']['cosine'] += \
            (1 - ((pred[i] * flow[i]).sum(dim=1) /
                  (pred_norm * flow_norm)).abs()).nanmean().item() / fraction
        report['stats']['error_metrics']['length'] += \
            (pred_norm - flow_norm).abs().mean().item() / fraction
        report['stats']['error_metrics']['l2-loss'] += \
            (pred[i] - flow[i]).norm(dim=1).mean().item() / fraction
    return report


def paint_pictures(vis_frames, sample, pred, flows, coords, dataset, label, paint_gt, report) :
    event_data = sample['in']['events']
    frame_i = sample['idx']
    if len(vis_frames) > 0:
        for i, frame_id in enumerate(sample['frame_id']):

            # TODO: fix this being a list when using dataloader
            if tuple(frame_id[:2]) in vis_frames:
                pred_i_np = pred[i].detach().cpu().numpy()
                # TODO: how to format image names
                im_name = (dataset.get_seq_name(frame_i[i]) + "/" +
                           str(frame_i[i]) + "_" +
                           dataset.get_frame_name(frame_i[i]))

                im = get_np_plot_flow(
                    create_event_picture(event_data[i], res=np.flip(dataset.res)),
                    coords[i],
                    pred_i_np,
                    flow_res=np.flip(dataset.res))
                report['ims'][im_name] = im

                im_color_name = (dataset.get_seq_name(frame_i[i]) + "/color_" +
                                 str(frame_i[i]) + "_" +
                                 dataset.get_frame_name(frame_i[i]))
                im_color = plot_flow_color(coords[i], pred_i_np, res=dataset.res)
                report['ims'][im_color_name] = im_color

                im_error_name = (dataset.get_seq_name(frame_i[i]) + "/error_" +
                                 str(frame_i[i]) + "_" +
                                 dataset.get_frame_name(frame_i[i]))
                im_error = plot_flow_error(coords[i], pred_i_np, flows[i].detach().cpu().numpy(),
                                           res=dataset.res)
                report['ims'][im_error_name] = im_error

                if paint_gt:
                    im_name_gt = (dataset.get_seq_name(frame_i[i]) + "/gt/" +
                                  str(frame_i[i]) + "_" +
                                  dataset.get_frame_name(frame_i[i]))
                    im_gt = get_np_plot_flow(
                        create_event_picture(event_data[i], res=np.flip(dataset.res)),
                        coords[i], flows[i].detach().cpu().numpy(),
                        flow_res=np.flip(dataset.res))

                    im_color_gt_name = (dataset.get_seq_name(frame_i[i]) + "/gt/color_" +
                                        str(frame_i[i]) + "_" +
                                        dataset.get_frame_name(frame_i[i]))
                    im_color_gt = plot_flow_color(coords[i], flows[i].detach().cpu().numpy(),
                                                  res=dataset.res)
                    report['ims'][im_color_gt_name] = im_color_gt

                    report['ims'][im_name_gt] = np.stack(im_gt)
    return report

def process_epoch(epoch, model, LFunc, dataset, device,
                  dataloader:DataLoader=None, optimizer=None, vis_frames=[],
                  full_query=False, predict_targets=False) :
    data = default(dataloader, dataset)
    report = {'stats' : {
                  'loss' : 0.,
                  'flow_length': {'pred': 0.,
                                  'gt': 0.},
                  'error_metrics' : {'cosine' : 0.,
                                     'length' : 0.,
                                     'l2-loss' : 0.}},
              'ims' : {}}

    for sample in data :
        if optimizer :
            optimizer.zero_grad(set_to_none=True)
        if not dataloader :
            sample = collate_dict_list([sample])

        batch_size = dataloader.batch_size if dataloader else 1

        out = forward_perceiver(model, sample, device, dataset, full_query, predict_targets)

        loss, pred, flows, coords = eval_perceiver_out(out, sample, device, LFunc, full_query)

        report['stats']['loss'] += loss.item() / len(dataset)

        if optimizer :
            loss /= batch_size
            loss.backward()
            optimizer.step()

        report = compute_statistics(pred, flows, report, len(dataset))

        report = paint_pictures(vis_frames, sample, pred, flows, coords, dataset,
                                label = "train" if optimizer else "val",
                                paint_gt=(epoch==0),
                                report=report)

    return report


def main() :
    args = parser.parse_args()

    save_figures_val = False
    save_figures_train = False
    save_figures_loss = False

    vis_train_frames = list(zip(*([iter(args.vis_train_frames)] * 2)))
    vis_val_frames = list(zip(*([iter(args.vis_val_frames)] * 2)))

    if hasattr(args, 'output_path') and args.output_path is not None :
        save_figures_val = True
        save_figures_train = False
        save_figures_loss = True
        output_path = Path(args.output_path)
        print(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(output_path / "settings.txt", 'w') as file :
            file.write("\n".join("{}: {}".format(k, v) for k, v in vars(args).items()))

    if hasattr(args, 'message') :
        print(args.message)

    perceiver_params = {}
    for i in range(len(args.perceiver_params) // 2) :
        perceiver_params[args.perceiver_params[2*i]] = int(args.perceiver_params[2*i+1])

    epochs = args.epochs
    lr = args.lr
    dt = args.dt
    batch_size = args.batch_size

    print_freq = 1
    val_freq = 2
    checkpoint_freq = args.checkpoint_freq


    torch.backends.cudnn.benchmark = True
    print("Benchmark: " + str(torch.backends.cudnn.benchmark))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    train_set = DSEC(seqs=args.train_seqs, dt=dt,
                     crop=args.crop,
                     random_crop_offset=args.random_crop_offset,
                     fixed_crop_offset=args.fixed_crop_offset,
                     random_moving=args.random_moving,
                     scale_crop=args.scale_crop,
                     crop_keep_full_res=args.crop_keep_full_res,
                     random_flip_horizontal=args.random_flip_horizontal,
                     random_flip_vertical=args.random_flip_vertical,
                     include_backward=args.include_backward,
                     num_bins=args.sum_groups,
                     event_set=args.event_set)

    # TODO: mulitple validation sets
    val_sets = [DSEC(seqs=args.val_seqs[i], dt=dt,
                     crop=args.crop if args.val_fixed_crop_offset else None,
                     random_crop_offset=False,
                     fixed_crop_offset=args.val_fixed_crop_offset,
                     random_moving=args.random_moving,
                     include_backward=args.include_backward,
                     num_bins=args.sum_groups,
                     event_set=args.event_set,
                     #,frames= [list(range(50))]
                     )
                for i in range(len(args.val_seqs))]


    print("Training set length: " + str(len(train_set)))
    # TODO: multiple validation sets
    for val_set in val_sets :
        print("Validation set length: " + str(len(val_set)))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = batch_size,
                                               collate_fn=collate_dict_list,
                                               num_workers=args.num_workers, shuffle=True,
                                               pin_memory=True)

    model = EventTransformer(pos_bands=args.xy_encoding_bands,
                             depth=args.depth,
                             res=train_set.res,
                             time_bands=args.t_encoding_bands,
                             perceiver_params=perceiver_params)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                          start_factor=args.warm_up_init,
                                                          total_iters=args.warm_up_length)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_halflife, gamma=0.5)

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

    # TODO: rename epch
    for it in range(epoch_0, epochs) :
        epoch_begin = time.time()
        if it == args.finetune_epoch :
            train_set.set_crop()

        report = process_epoch(it, model, LFunc, train_set, device,
                               train_loader, optimizer,
                               vis_train_frames if it % args.vis_freq == 0 else [],
                               args.full_query, args.predict_targets)

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


        if it % val_freq == 0 or it % args.vis_freq == 0 :
            model.eval()

            for idx_val_set, val_set in enumerate(val_sets) :
                with torch.no_grad() :
                    report = process_epoch(it, model, LFunc, val_set, device,
                                           vis_frames=vis_val_frames
                                           if it % args.vis_freq == 0 else [],
                                           full_query=args.full_query,
                                           predict_targets=args.predict_targets)

                    # TODO: print frame
                print("Iteration: " + str(it) + ", validation loss : " +
                      "{0:.3f}".format(report['stats']['loss']))
                val_loss_histories[idx_val_set].append(report['stats']['loss'])

                write_stats_to_tensorboard(report['stats'], writer, it, prefix="val"+str(idx_val_set))

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
