import torch
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError


class TrainingModule(pl.LightningModule) :
    def __init__(self, model, model_trainer) :
        super().__init__()
        self.model = model
        self.model_trainer = model_trainer
        self.sample_count = 0
        self.epoch_count = 0

        self.loss_running_average = MeanAbsoluteError()
        self.losses = []

        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def forward(self, x) :
        return self.model_trainer.forward(self.model, x)

    def training_step(self, batch, batch_idx) :
        batch_size = self.model_trainer.batch_size(batch)
        out = self(batch)
        eval = self.model_trainer.evaluate(batch, out)

        stats = self.model_trainer.statistics(batch, out, eval)

        self.sample_count += batch_size
        self.loss_running_average(eval['loss'], torch.zeros(eval['loss'].shape,
                                                                   device=eval['loss'].device))
        self.losses.append(eval['loss'].detach())

        self.log('step', self.sample_count)
        self.log('train/loss', eval['loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        return eval

    def training_epoch_end(self, outputs) :
        """
        self.log('step', self.epoch_count)
        self.log('train_epoch/loss',
                 torch.cat([o['loss'].reshape(-1) for o in outputs]).mean(),
                 on_epoch=True, on_step=False)
        """
        self.epoch_count += 1
        print("Custom loss aggregation")
        print(torch.cat([l.reshape(-1) for l in self.losses]).mean())
        for param_group in self.opt.param_groups:
            print(param_group['lr'])

        self.losses = []

    def validation_step(self, batch, batch_idx) :
        loss = self.model_trainer.evaluate(batch, self(batch))['loss']
        self.log('step', self.sample_count)
        self.log('val0/loss', loss)
        return loss

    """
    def training_step_end(self, outputs) :
        print(self.sample_count)
        print(outputs['loss'])
        print("training step end")
        tensorboard_logs = {'test' : 1337, 'step': 20}

        return {'loss' : torch.mean(outputs['loss']),
                'log': tensorboard_logs}
    """

    def configure_optimizers(self) :
        print("configure optimizers")
        # TODO: how to feed learning rate?
        return self.opt


# TODO: callback for every n samples seen


class DataModule(pl.LightningDataModule) :
    def train_dataloader(self) :
        pass