import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader

import time
import random

import heapq

from training.training_report import *
from utils import default, nested_to_device
from data.utils import collate_dict_list

from training.training_interface import *



def config_add_defaults(config) :
    config['training'].setdefault('params', {})

    config['data_loader'].setdefault('shuffle', True)
    config['data_loader'].setdefault('pin_memory', True)

    config['output'].setdefault('checkpoint_freq', 1000)

    return config


class TrainerTraining :
    def __init__(self, model, training_set, validation_sets,
                 model_trainer:AbstractTrainer,
                 config, device, output_path, collate_fn=None) :
        self.config = config_add_defaults(config)
        self.model = model

        self.train_set = training_set

        self.val_sets = validation_sets

        print("Training set length: " + str(len(self.train_set)))
        for val_set in self.val_sets:
            print("Validation set length: " + str(len(val_set)))

        self.model_trainer = model_trainer

        self.device = device

        self.output_path = output_path

        def set_worker_sharing_strategy(worker_id: int) -> None:
            torch.multiprocessing.set_sharing_strategy('file_system')

        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        collate_fn=collate_fn,
                                                        **self.config['data_loader'],
                                                        worker_init_fn=set_worker_sharing_strategy)

        self.optimizer = torch.optim.Adam(model.parameters(), config['training']['lr'])

        self.schedulers = []
        self.schedulers_on_epoch_fraction = []
        self.schedulers_on_samples = []
        self.schedulers_on_val_loss = []

        if 'lr_halflife' in config['training'] :
            print("code not maintained")
            scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config['training']['lr_halflife'],
                    gamma=0.5,
                    verbose=True)
            self.schedulers.append(scheduler)
        if 'warm_up' in config['training'] :
            print("code not maintained")
            self.schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=config['training']['warm_up']['init'],
                    total_iters=config['training']['warm_up']['length'],
                    verbose=True))


        if 'reduce_lr_on_plateau' in config['training'] :
            print("code not maintained")
            self.schedulers_on_val_loss.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    verbose=True,
                    **config['training']['reduce_lr_on_plateau']
                ))
        if 'cosine_annealing_warm_restarts' in config['training'] :
            self.schedulers_on_epoch_fraction.append(
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    verbose=True,
                    **config['training']['cosine_annealing_warm_restarts']
                )
            )

        self.epoch_count = 0
        self.step_count = 0
        self.sample_count = 0

        self.running_train_report = self.empty_report()




        self.running_report = self.empty_report()
        self.running_report['runtime']['total'] = time.time()
        self.train_last_report_samples_seen = 0

        self.writer = tensorboard.SummaryWriter(log_dir=output_path) #, purge_step=epoch_0)


        self.train_loss_history = []
        self.train_loss_steps = []

        self.val_loss_histories = [[] for _ in self.val_sets]
        self.val_loss_steps = []

        self.output_on_samples = config['output']['on_samples']

        self.eval_val_epoch_freq = 2
        self.vis_val_epoch_freq = config['output']['vis_epoch_freq']
        self.eval_val_sample_freq = None
        self.vis_val_sample_freq = None
        self.fail_val_sample_freq = None


        self.eval_train_sample_freq = len(self.train_set)
        self.fail_train_sample_freq = None

        self.vis_train_epoch_freq = config['output']['vis_epoch_freq']

        self.val_last_report_samples_seen = 0

        # TODO: incorporate these into
        if 'eval_val_sample_freq' in config['output'] :
            self.eval_val_sample_freq = config['output']['eval_val_sample_freq']
        if 'vis_val_sample_freq' in config['output'] :
            self.vis_val_sample_freq = config['output']['vis_val_sample_freq']
        if 'fail_val_sample_freq' in config['output'] :
            self.fail_val_sample_freq = config['output']['fail_val_sample_freq']

        if 'eval_train_sample_freq' in config['output'] :
            self.eval_train_sample_freq = config['output']['eval_train_sample_freq']
        if 'fail_train_sample_freq' in config['output'] :
            self.fail_train_sample_freq = config['output']['fail_train_sample_freq']

        self.vis_val_frames = config['output']['vis_val_frames']
        self.vis_train_frames = config['output']['vis_train_frames']

        self.checkpoint_freq = config['output']['checkpoint_freq']


    def update_failcases(self, energies, item, out, eval, queue, max_length=20) :
        for i in range(len(energies)) :
            if len(queue) < max_length or energies[i] > queue[0][0]  :
                if len(queue) >= max_length : heapq.heappop(queue)
                d = self.model_trainer.visualize_item(i, item, out, eval, True)
                heapq.heappush(queue,
                               (energies[i],
                                d))
        return queue


    def training_sanity_check(self) :
        batch_size = self.train_loader.batch_size
        print(batch_size)
        batch_list = [self.train_set[i] for i in range(batch_size)]
        batch = self.train_set.collate(batch_list)
        batch_device = self.model_trainer.sample_to_device(batch, self.device)
        self.model.train()
        out = self.model_trainer.forward(self.model, batch_device)
        eval = self.model_trainer.evaluate(batch_device,  out)
        eval['loss'].backward()
        print(torch.cuda.memory_allocated(device=self.device) / 1024**3)
        print(torch.cuda.memory_reserved(device=self.device) / 1024**3)
        print(torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 3)
        print(torch.cuda.max_memory_reserved(device=self.device) / 1024 ** 3)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()


    def process_item(self, item, data, optimize=False, vis_frames=[], report=None) :
        if report is None : report = self.empty_report()
        time_begin = time.time()
        if type(data) is torch.utils.data.DataLoader:
            batch_size = self.model_trainer.batch_size(item)
        else:
            batch_size = 1
            item = data.collate([item])

        if optimize:
            self.step_count += 1
            self.sample_count += batch_size
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()

        t_sample_to_device_begin = torch.cuda.Event(enable_timing=True)
        t_sample_to_device_end = torch.cuda.Event(enable_timing=True)
        t_sample_to_device_begin.record()
        sample_device = self.model_trainer.sample_to_device(item, self.device)
        t_sample_to_device_end.record()

        forward_begin = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_begin.record()
        out = self.model_trainer.forward(self.model, sample_device)
        forward_end.record()

        loss_begin = torch.cuda.Event(enable_timing=True)
        loss_end = torch.cuda.Event(enable_timing=True)
        loss_begin.record()
        eval = self.model_trainer.evaluate(sample_device, out)
        loss_end.record()

        loss = eval['loss']

        backward_begin = torch.cuda.Event(enable_timing=True); backward_begin.record()
        if optimize:
            loss.backward()
            self.optimizer.step()
        backward_end = torch.cuda.Event(enable_timing=True); backward_end.record()

        t_stats_begin = torch.cuda.Event(enable_timing=True); t_stats_begin.record()
        stats = self.model_trainer.statistics(item, out, eval, 'failcases' in report)
        t_stats_end = torch.cuda.Event(enable_timing=True); t_stats_end.record()


        images_begin = time.time()
        report['ims'].update(self.model_trainer.visualize(vis_frames, item, out, eval))
        report['runtime']['images'] = time.time() - images_begin


        if 'failcases' in report :
            report['failcases'] = self.update_failcases([s['point_errors']['epe'] / s['num_points'] for s in stats],
                                                        sample_device,
                                                        out,
                                                        eval,
                                                        report['failcases'])
            stats = sum_stats_list(stats)


        stats['loss'] = loss.detach() * batch_size

        #t_stats_begin.synchronize()
        #report['runtime']['stats'] = t_stats_begin.elapsed_time(t_stats_end)

        # torch.cuda.synchronize()
        #report['runtime']['sample_to_device'] += t_sample_to_device_begin.elapsed_time(t_sample_to_device_end) / 1000
        #report['runtime']['forward'] += forward_begin.elapsed_time(forward_end) / 1000
        #if 'torch_cuda_events' in out :
        #    report['runtime']['model'] += out['torch_cuda_events'][0].elapsed_time(out['torch_cuda_events'][1]) / 1000
        #report['runtime']['loss'] += loss_begin.elapsed_time(loss_end) / 1000
        #report['runtime']['backward'] += backward_begin.elapsed_time(backward_end) / 1000

        report['stats'] = sum_stats(report['stats'], stats)
        report['num_samples'] += batch_size

        report['runtime']['batch_total'] += time.time() - time_begin

        return report


    def empty_report(self) :
        return {
            'num_samples' : 0,
            'stats': {
                'loss': 0.,
                'flow_length': {'pred': 0.,
                                'gt': 0.},
                'error_metrics': {'angle': 0.,
                                  'length': 0.,
                                  'l2-loss': 0.},
                'num_points': 0,
                'point_errors'  : {'epe' : 0.,
                                   '1pe' : 0.,
                                   '2pe' : 0.,
                                   '3pe' : 0.}},
            'runtime': {'total' : 0.,
                        'batch_total': 0.,
                        'sample_to_device' : 0.,
                        'forward': 1e-8,
                        'model': 0.,
                        'loss': 0.,
                        'backward': 0.,
                        'stats': 0.,
                        'images': 0.},
            'ims': {}}


    def process_data(self, data, optimize=False, vis_frames=[], track_failcases=False):
        epoch_begin = time.time()
        report = self.empty_report()
        if track_failcases : report['failcases'] = []

        for sample in data:
            report = self.process_item(sample, data, optimize, vis_frames, report)

        report['runtime']['total'] = time.time() - epoch_begin
        return report


    def process_validation(self, vis_frames:bool, track_failcases=False) :
        self.model.eval()
        losses = []
        for idx_val_set, val_set in enumerate(self.val_sets):
            with torch.no_grad():
                report = self.process_data(val_set,
                                           vis_frames=
                                           self.vis_val_frames
                                           if vis_frames else [],
                                           track_failcases=track_failcases)

                self.write_report(report, "val", idx_val_set, self.output_on_samples)
                losses.append(report['stats']['loss'].item())
        return losses

    def process_validation_for_epoch_if_required(self) :
        if (self.epoch_count % self.eval_val_epoch_freq == 0 or
            self.epoch_count % self.vis_val_epoch_freq == 0) :
            self.process_validation(self.epoch_count % self.vis_val_epoch_freq == 0,
                                    False)

    def process_validation_for_sample_if_required(self) :
        if (self.eval_train_sample_freq is not None and self.eval_val_sample_freq is not None and
                self.sample_count - self.val_last_report_samples_seen >= self.eval_val_sample_freq) :
            self.val_last_report_samples_seen += self.eval_val_sample_freq
            losses = \
            self.process_validation(self.vis_val_sample_freq is not None and
                                    self.val_last_report_samples_seen % self.vis_val_sample_freq <
                                    self.eval_val_sample_freq,
                                    self.fail_val_sample_freq is not None and
                                    self.val_last_report_samples_seen % self.fail_val_sample_freq <
                                    self.eval_val_sample_freq)
            for scheduler in self.schedulers_on_val_loss :
                scheduler.step(losses[0])


    def run_epoch(self) :
        self.process_validation_for_epoch_if_required()
        t_epoch = time.time()
        report = self.empty_report()
        vis_frames = (self.vis_train_frames
                      if self.vis_train_epoch_freq and
                         self.epoch_count % self.vis_train_epoch_freq == 0
                      else [])

        for i, batch in enumerate(self.train_loader) :
            batch_report = self.process_item(batch, self.train_loader, True,
                                             vis_frames)
            write_ims_to_tensorboard(batch_report['ims'], self.writer,
                                     self.sample_count if self.output_on_samples else self.epoch_count,
                                     "train")
            batch_report['ims'] = {}
            report = sum_stats(report, batch_report)

            if self.output_on_samples:
                self.running_report = sum_stats(self.running_report, batch_report)

                if self.sample_count - self.train_last_report_samples_seen >= self.eval_train_sample_freq:
                    self.running_report['lr'] = self.optimizer.param_groups[0]['lr']
                    self.running_report['runtime']['total'] = time.time() - self.running_report['runtime']['total']
                    self.write_report(self.running_report, "train", output_on_samples=True)
                    self.running_report = self.empty_report()
                    self.train_last_report_samples_seen +=self.eval_train_sample_freq

                    self.process_validation_for_sample_if_required()
                    self.running_report['runtime']['total'] = time.time()
                else :
                    self.process_validation_for_sample_if_required()

            for scheduler in self.schedulers_on_samples : scheduler.step(self.sample_count)
            for scheduler in self.schedulers_on_epoch_fraction : scheduler.step(self.epoch_count + i / len(self.train_loader))


        report['lr'] = self.optimizer.param_groups[0]['lr']
        report['runtime']['total'] = time.time() - t_epoch
        self.write_report(report, "train" if not self.output_on_samples else "train_epoch")
        self.epoch_count += 1
        for scheduler in self.schedulers: scheduler.step(self.epoch_count)
        for scheduler in self.schedulers_on_epoch_fraction: scheduler.step(self.epoch_count)
        print("GPU memory allocated : " + str(torch.cuda.max_memory_allocated(device=self.device) / 1024**3))



        if self.epoch_count % self.checkpoint_freq == 0 :
            self.write_checkpoint()


    def set_finetune(self) :
        raise NotImplementedError()


    def write_report(self, report, label, idx=None, output_on_samples=False) :
        if output_on_samples :
            counter = self.sample_count
        else :
            counter = self.epoch_count
        if label == "train" :
            self.train_loss_history.append(report['stats']['loss'])
            self.train_loss_steps.append(counter)
        elif label == "val" :
            self.val_loss_histories[idx].append(report['stats']['loss'])
            label = label + str(idx)
            if idx == 0 : self.val_loss_steps.append(counter)


        if 'stats' in report :
            print("Report num samples: " + str(report['num_samples']))
            write_stats_to_tensorboard(report['stats'],
                                       self.writer, counter, prefix=label,
                                       fraction=report['num_samples'])
        if 'lr' in report :
            write_stats_to_tensorboard({'lr' : report['lr']},
                                       self.writer, counter, prefix=label,
                                       fraction=1.)

        if 'ims' in report :
            write_ims_to_tensorboard(report['ims'], self.writer, counter, prefix=label)

        if 'runtime' in report :
            write_runtime_stats_to_tensorboard(report['runtime'], self.writer, counter, prefix=label,
                                               samples=report['num_samples'] if 'stats' in report else 1)

        if 'failcases' in report :
            queue = report['failcases']
            im_grid = [list(q[1].values()) for q in queue]
            im = create_image_grid(im_grid)
            self.writer.add_image('failcases/' + label,
                                  im.transpose((2, 0, 1)), counter)


        it_string = ("Epoch: " + str(self.epoch_count) +
                     ", Step: " + str(self.step_count) +
                     ", Sample: " + str(self.sample_count) +
                     ", " + label + " loss : " +
                     "{0:.3f}".format(report['stats']['loss'] / report['num_samples']) +
                     ", time: " + "{0:.1f}".format(report['runtime']['total']))
        print(it_string)
        with open(self.output_path / "output.txt", 'a') as file:
            file.write(it_string + "\n")


    def load_checkpoint(self, path, model_only=False, load_schedulers=True) :
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if not model_only :
            self.epoch_count = checkpoint['epoch']
            self.step_count = checkpoint['step']
            self.sample_count = checkpoint['sample']
            self.running_report = checkpoint['running_report']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if load_schedulers :
                for i in range(min(len(self.schedulers), len(checkpoint['scheduler']))) :
                    self.schedulers[i].load_state_dict(checkpoint['scheduler'][i])
                if 'scheduler_on_samples' in checkpoint :
                    for i in range(min(len(self.schedulers_on_samples), len(checkpoint['scheduler_on_samples']))):
                        self.schedulers_on_samples[i].load_state_dict(checkpoint['scheduler_on_samples'][i])
                if 'scheduler_on_val_loss' in checkpoint :
                    for i in range(min(len(self.schedulers_on_val_loss), len(checkpoint['scheduler_on_val_loss']))):
                        self.schedulers_on_val_loss[i].load_state_dict(checkpoint['scheduler_on_val_loss'][i])
                if 'scheduler_on_epoch_fraction' in checkpoint :
                    for i in range(min(len(self.schedulers_on_epoch_fraction), len(checkpoint['scheduler_on_epoch_fraction']))):
                        self.schedulers_on_epoch_fraction[i].load_state_dict(checkpoint['scheduler_on_epoch_fraction'][i])
            self.train_loss_history = checkpoint['train_loss_history']
            self.train_loss_steps = checkpoint['train_loss_steps']
            self.val_loss_histories = checkpoint['val_loss_histories']
            self.val_loss_steps = checkpoint['val_loss_steps']
            self.train_last_report_samples_seen = checkpoint['train_last_report_samples_seen']
            self.val_last_report_samples_seen = checkpoint['val_last_report_samples_seen']
            random.setstate(checkpoint['py_rng'])
            torch.set_rng_state(checkpoint['torch_rng'])


    def write_checkpoint(self) :
        checkpoint = {
            'epoch': self.epoch_count,
            'step' : self.step_count,
            'sample' : self.sample_count,
            'running_report' : self.running_report,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': [scheduler.state_dict()
                          for scheduler in self.schedulers],
            'scheduler_on_samples' : [scheduler.state_dict()
                                      for scheduler in self.schedulers_on_samples],
            'scheduler_on_val_loss' : [scheduler.state_dict()
                                       for scheduler in self.schedulers_on_val_loss],
            'scheduler_on_epoch_fraction' : [scheduler.state_dict()
                                             for scheduler in self.schedulers_on_epoch_fraction],
            'train_loss_history': self.train_loss_history,
            'train_loss_steps' : self.train_loss_steps,
            'val_loss_histories': self.val_loss_histories,
            'val_loss_steps': self.val_loss_steps,
            'train_last_report_samples_seen' : self.train_last_report_samples_seen,
            'val_last_report_samples_seen' : self.val_last_report_samples_seen,
            'py_rng': random.getstate(),
            'torch_rng': torch.get_rng_state()
        }
        torch.save(checkpoint, self.output_path / ("checkpoint" + str(self.epoch_count)))


