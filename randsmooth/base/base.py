import torch
import pytorch_lightning as pl


class Base(pl.LightningModule):
    """A base prediction model as a lightning module.

    Children need to override forward and self.loss_func.
    """

    def __init__(self, data=None):
        super().__init__()
        self.data = data
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        assert self.training

        return self.compute_losses(batch, 'loss')

    def validation_step(self, batch, batch_idx):
        assert not self.training

        with torch.no_grad():
            return self.compute_losses(batch, 'val_loss')

    def test_step(self, batch, batch_idx):
        assert not self.training

        with torch.no_grad():
            return self.compute_losses(batch, 'test_loss')

    def extra_log(self, experiment, outputs):
        pass

    def compute_losses(self, batch, loss_string):
        signal, target = batch[0], batch[1]
        preds = self.forward(signal)

        vars = {}
        vars['signal'], vars['target'] = signal, target
        vars['preds'] = preds.detach()
        vars['acc'] = self.compute_accuracy(preds.detach(), target)
        vars['loss'] = self.loss_func(preds, target)
        vars[loss_string] = vars['loss']

        # For the model checkpointing
        self.log(loss_string, vars[loss_string])

        return vars

    def compute_accuracy(self, pred, target):
        return self.get_correct_entries(pred, target).sum().double() / pred.shape[0]

    def get_correct_entries(self, pred, target):
        _, pred_class = torch.max(pred, 1)
        return (pred_class == target)

    def training_epoch_end(self, outputs):
        self.log_outputs(outputs, 'Train')

        self.extra_log(self.logger.experiment, outputs)

        self.logger.experiment.flush()

    def validation_epoch_end(self, outputs):
        for epoch_out in outputs:
            epoch_out['loss'] = epoch_out.pop('val_loss')

        self.log_outputs(outputs, 'Valid')

    def log_outputs(self, outputs, type_string):
        experiment = self.logger.experiment
        losses = self.calc_means(outputs, 'loss')
        experiment.add_scalars(f"Loss/{type_string}", losses, self.current_epoch)

        accs = self.calc_means(outputs, 'acc')
        experiment.add_scalars(f"Accuracies/{type_string}", accs, self.current_epoch)

    def calc_means(self, outputs, in_key):
        return {key: self.calc_mean(outputs, key)
                for key in outputs[0].keys() if in_key in key}

    def calc_mean(self, outputs, key):
        return torch.stack([x[key] for x in outputs]).mean()
