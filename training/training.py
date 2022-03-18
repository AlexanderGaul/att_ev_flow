import numpy as np

import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader

import time
import random

from utils import default, collate_dict_list, nested_to_device
from plot import *



def compute_statistics(pred, flow, report, fraction) :
    for i in range(len(pred)):
        if len(pred[i]) == 0:
            continue
        pred_norm = pred[i].norm(dim=1)
        flow_norm = flow[i].norm(dim=1)
        report['stats']['flow_length']['pred'] += \
            pred_norm.nanmean().item() / fraction
        report['stats']['flow_length']['gt'] += \
            flow_norm.nanmean().item() / fraction
        report['stats']['error_metrics']['cosine'] += \
            (1 - ((pred[i] * flow[i]).sum(dim=1) /
                  (pred_norm * flow_norm)).abs()).nanmean().item() / fraction
        report['stats']['error_metrics']['length'] += \
            (pred_norm - flow_norm).abs().nanmean().item() / fraction
        report['stats']['error_metrics']['l2-loss'] += \
            (pred[i] - flow[i]).norm(dim=1).nanmean().item() / fraction
    return report


def paint_pictures(E_im, coords, pred, flows, label, report) :
    im = get_np_plot_flow(
        E_im,
        coords,
        pred,
        flow_res=E_im.shape[:2])
    report['ims'][label] = im

    im_gt = get_np_plot_flow(
        E_im,
        coords,
        flows,
        flow_res=E_im.shape[:2])
    report['ims'][label + "/gt"] = im_gt

    im_color = plot_flow_color(coords, pred, res=np.flip(E_im.shape[:2]))
    report['ims'][label + "/color"] = im_color

    im_error = plot_flow_error(coords, pred, flows,
                               res=np.flip(E_im.shape[:2]))
    report['ims'][label + "/error"] = im_error

    return report


def visualize(vis_frames, sample, coords, pred, flows, report) :
    # TODO: make sample into batch before calling this
    if len(vis_frames) == 0 :
        return report
    is_batched = hasattr(sample['frame_id'][0], '__len__')

    if 'events' in sample.keys() :
        events = sample['events']
    else :
        events = sample['event_volume_new']

    assert is_batched
    """    if not is_batched :
        if sample['frame_id'][:2] in vis_frames :
            report = paint_pictures(
                create_event_picture(events.detach().cpu().numpy(),
                                     res=np.flip(sample['res']) if 'res' in sample
                                    else events.shape[1:3]),
                coords[0].detach().cpu().numpy(), pred[0].detach().cpu().numpy(), flows[0].detach().cpu().numpy(),
                str(sample['frame_id'][0]) + "_" + str(sample['frame_id'][1]),
                report)
        return report"""

    for i, id in enumerate(sample['frame_id']) :
        if list(id[:2]) in vis_frames :
            report = paint_pictures(
                create_event_picture(events[i].detach().cpu().numpy(),
                                     res=np.flip(sample['res'][0]) if 'res' in sample
                                     else events[i].shape[1:3]),
                coords[i].detach().cpu().numpy(), pred[i].detach().cpu().numpy(), flows[i].detach().cpu().numpy(),
                str(sample['frame_id'][i]),
                report)

    return report


def process_epoch(epoch, model, LFunc, dataset, device,
                  forward_model_fun, eval_model_out_fun,
                  dataloader:DataLoader=None, optimizer=None, vis_frames=[],
                  batch_size=1,
                  **kwargs) :
    data = default(dataloader, dataset)
    report = {'stats' : {
                  'loss' : 0.,
                  'flow_length': {'pred': 0.,
                                  'gt': 0.},
                  'error_metrics' : {'cosine' : 0.,
                                     'length' : 0.,
                                     'l2-loss' : 0.}},
                  'runtime': {'total': 0.,
                              'forward': 0.,
                              'model': 0.,
                              'loss': 0.,
                              'backward': 0.,
                              'stats' : 0.,
                              'images' : 0.},
              'ims' : {}}

    for sample in data :
        time_begin = time.time()
        if optimizer :
            optimizer.zero_grad(set_to_none=True)

        if dataloader is None :
            sample = collate_dict_list([sample])

        sample_device = nested_to_device(sample, device)

        forward_begin = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_begin.record()
        out = forward_model_fun(model, sample_device, dataset, **kwargs)
        forward_end.record()

        loss_begin = torch.cuda.Event(enable_timing=True)
        loss_end = torch.cuda.Event(enable_timing=True)
        loss_begin.record()
        loss, pred, flows, coords = eval_model_out_fun(out, sample_device, LFunc, **kwargs)
        loss_end.record()

        backward_begin = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        backward_begin.record()
        if optimizer :
            loss /= batch_size
            loss *= batch_size ** 0.5
            loss.backward()
            optimizer.step()
        backward_end.record()

        stats_begin = time.time()
        report = compute_statistics(pred, flows, report, len(dataset))
        report['runtime']['stats'] = time.time() - stats_begin

        images_begin = time.time()
        report = visualize(vis_frames, sample, coords, pred, flows, report)
        report['runtime']['images'] = time.time() - images_begin

        torch.cuda.synchronize()
        report['runtime']['forward'] += forward_begin.elapsed_time(forward_end)
        #report['runtime']['model'] += fwd_call_ev[0].elapsed_time(fwd_call_ev[1])
        report['runtime']['loss'] += loss_begin.elapsed_time(loss_end)
        report['runtime']['backward'] += backward_begin.elapsed_time(backward_end)
        report['stats']['loss'] += loss.item() / (batch_size ** 0.5) / len(dataset) * batch_size
        report['runtime']['total'] += time.time() - time_begin

    return report


def config_add_defaults(config) :
    config['training'].setdefault('params', {})

    config['data_loader'].setdefault('shuffle', True)
    config['data_loader'].setdefault('pin_memory', True)

    config['output'].setdefault('checkpoint_freq', 1000)

    return config


# could move collate function to dataste which knows what to combine 

class Training :
    def __init__(self, model, DataSetClass, forward_model_func, eval_model_func, loss_func,
                 config, device, output_path, collate_fn=None) :
        self.config = config_add_defaults(config)


        self.model = model

        self.train_set = DataSetClass(**self.config['train_set'])

        self.val_sets = [DataSetClass(**self.config['val_sets'][i])
                         for i in range(len(config['val_sets']))]

        print("Training set length: " + str(len(self.train_set)))
        for val_set in self.val_sets:
            print("Validation set length: " + str(len(val_set)))

        self.forward_model_func = forward_model_func
        self.eval_model_func = eval_model_func
        self.loss_func = loss_func

        self.device = device

        self.output_path = output_path

        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        collate_fn=collate_fn,
                                                        **self.config['data_loader'])

        self.optimizer = torch.optim.Adam(model.parameters(), config['training']['lr'])

        self.schedulers = []
        if 'lr_halflife' in config['training'] :
            self.schedulers.append(
                torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config['training']['lr_halflife'],
                    gamma=0.5))
        if 'warm_up' in config['training'] :
            self.schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=config['training']['warm_up']['init'],
                    total_iters=config['training']['warm_up']['length']))

        self.epoch = 0

        self.batch_size = self.train_loader.batch_size
        # TOOD: streamline this
        if type(self.train_set[0]) is list :
            self.batch_size *= len(self.train_set[0])

        self.writer = tensorboard.SummaryWriter(log_dir=output_path) #, purge_step=epoch_0)

        self.train_loss_history = []
        self.val_loss_histories = [[] for _ in self.val_sets]
        self.val_loss_steps = []

        self.val_freq = 2


    def __del__(self) :
        self.writer.flush()
        self.writer.close()

    def step(self) :
        epoch_begin = time.time()
        """if it == args.finetune_epoch :
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
            train_set.random_backward = False"""

        report = process_epoch(self.epoch, self.model, self.loss_func, self.train_set,
                               self.device,
                               self.forward_model_func, self.eval_model_func,
                               self.train_loader, self.optimizer,
                               self.config['output']['vis_train_frames'] if self.epoch % self.config['output']['vis_freq'] == 0 else [],
                               batch_size=self.batch_size,
                               **self.config['training']['params'])

        for scheduler in self.schedulers :
            scheduler.step()

        write_stats_to_tensorboard(report['stats'], self.writer, self.epoch, prefix="train")
        self.train_loss_history.append(report['stats']['loss'])

        # im_grid = torchvision.utils.make_grid([torch.tensor(im.transpose((2, 0,1 )))
        #                                        for im in stats['ims'].values()])
        for im_name in report['ims'].keys() :
            self.writer.add_image("train/" + im_name,
                                  report['ims'][im_name].transpose((2, 0, 1)), self.epoch)

        it_string = ("Iteration: " + str(self.epoch) + ", training loss : " +
              "{0:.3f}".format(report['stats']['loss']) +
              ", time: " + "{0:.1f}".format(time.time() - epoch_begin))
        print(it_string)
        with open(self.output_path / "output.txt", 'w') as file:
            file.write(it_string)


        if self.epoch % self.val_freq == 0 or self.epoch % self.config['output']['vis_freq'] == 0 :
            self.model.eval()

            for idx_val_set, val_set in enumerate(self.val_sets) :
                with torch.no_grad() :
                    report = process_epoch(self.epoch, self.model, self.loss_func, val_set,
                                           self.device,
                                           self.forward_model_func, self.eval_model_func,
                                           vis_frames=self.config['output']['vis_val_frames']
                                           if self.epoch % self.config['output']['vis_freq'] == 0 else [],
                                           **self.config['training']['params'])

                    # TODO: print frame
                print("Iteration: " + str(self.epoch) + ", validation loss : " +
                      "{0:.3f}".format(report['stats']['loss']))
                self.val_loss_histories[idx_val_set].append(report['stats']['loss'])

                write_stats_to_tensorboard(report['stats'], self.writer, self.epoch, prefix="val" + str(idx_val_set))

                for im_name in report['ims'].keys():
                    self.writer.add_image("val" + str(idx_val_set) + "/" + im_name,
                                     report['ims'][im_name].transpose((2, 0, 1)), self.epoch)

            self.val_loss_steps.append(self.epoch)


            """if save_figures_loss :
                plt.plot(np.arange(0, len(train_loss_history)), train_loss_history, label="train")
                for i, val_loss_history in enumerate(val_loss_histories) :
                    if len(val_loss_history) == len(val_loss_steps):
                        plt.plot(val_loss_steps, val_loss_history, label="val"+str(i))
                plt.legend(loc='upper right')
                plt.savefig(output_path / "loss_history.png",
                            pad_inches=0.)
                plt.close()"""

            self.model.train()

        self.epoch += 1

        if self.epoch % self.config['output']['checkpoint_freq'] == 0 :
            self.write_checkpoint()


    def set_finetune(self) :
        pass


    def load_checkpoint(self, path) :
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for i, scheduler in enumerate(self.schedulers) :
            scheduler.load_state_dict(checkpoint['scheduler'][i])
        self.train_loss_history = checkpoint['train_loss_history']
        self.val_loss_histories = checkpoint['val_loss_histories']
        self.val_loss_steps = checkpoint['val_loss_steps']
        random.setstate(checkpoint['py_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])


    def write_checkpoint(self) :
        checkpoint = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': [scheduler.state_dict()
                          for scheduler in self.schedulers],
            'train_loss_history': self.train_loss_history,
            'val_loss_histories': self.val_loss_histories,
            'val_loss_steps': self.val_loss_steps,
            'py_rng': random.getstate(),
            'torch_rng': torch.get_rng_state()
        }
        torch.save(checkpoint, self.output_path / ("checkpoint" + str(self.epoch)))



def write_stats_to_tensorboard(stats:dict, writer:tensorboard.SummaryWriter,
                               it:int,
                               prefix:str="", suffix:str="") :
    for k in stats.keys() :
        if type(stats[k]) is dict :
            write_stats_to_tensorboard(stats[k], writer, it,
                                       prefix=prefix+"/"+k, suffix=suffix)
        else :
            writer.add_scalar(prefix+"/"+k+"/"+suffix, stats[k], it)