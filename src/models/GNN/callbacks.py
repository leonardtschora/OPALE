from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np

class EarlyStoppingSlidingAverage(EarlyStopping):
    def __init__(self, monitor="val_loss", verbose=False, alpha=10, patience=10,
                 restore_best_weights=False):
        EarlyStopping.__init__(self, monitor=monitor, verbose=verbose,
                               patience=patience)
        self.alpha = alpha
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.val_losses = []
        self.best = 10e9
        self.best_weights = None
        
    def on_validation_end(self, trainer, pl_module):
        current = self.get_monitor_value(trainer.callback_metrics)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights and verbose > 0:
                print("Can't restore weights of a torch module")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print("Can't restore weights of a torch module")
        
    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor).detach().numpy()
        if monitor_value is None:
            print(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        self.val_losses.append(monitor_value.ravel()[0])
        try:
            current = len(self.val_losses)
        except:
            current = 0
            values = [self.val_losses[current]]
        else:
            k = min(self.alpha, current)
            values = self.val_losses[current - k:current]
            
        mean_ = np.mean(values)
        return mean_
