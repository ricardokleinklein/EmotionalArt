"""
Main training tools, wrapped around a single Trainer object.

The Trainer interfase allows its usage either independently or within the
context of a bigger experimentation pipeline. It implements both the
training and testing utilities, so it doesn't matter whether you're just
looking for a system to evaluate your pretrained model, go through the
Trainer class for a smooth experience.

TODO: Swap to a cleaner LOG system.
TODO: Enable changing the optimizer.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from typing import Callable, Dict, Optional, Union, Tuple

import loggers.baselog
from metrics.base_metrics import Metrics

Loader = torch.utils.data.DataLoader
Tensor = torch.Tensor


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 loss_fn: Union[Callable, nn.Module],
                 metrics: Metrics,
                 monitor_metric: str,
                 device: torch.device,
                 learning_rate: float = 1e-5,
                 logger: Optional[loggers.baselog.Logger] = None,
                 *args, **kwargs):
        """ Systematize a typical lifecycle of a machine learning
        pipeline, including training, validating and testing one.

        TODO: Switch to model checkpoint.

        Args:
            model: Torch model to work with.
            loss_fn: Error function.
            metrics: Metrics to assess the training and evaluation by.
            monitor_metric: Monitor to ponder performance.
            device: Which device to use (CPU / GPU).
            logger: Initialized logger on which to dump training logs.
        """
        self.model = copy.deepcopy(model).to(device)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.monitor_metric = monitor_metric
        self.device = device
        self.logger = logger

        self.best_model = copy.deepcopy(model)
        self.best_loss = 0
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

    def fit(self, data_loader: Loader, val_loader: Optional[Loader] = None,
            max_epochs: int = 500, patience: int = 5, verbose: bool = True,
            tol_eps: float = 0.0) -> Dict:
        """ Automatic pipeline to train a model.

        Args:
            data_loader: Training data, ust contain inputs and labels.
            max_epochs: Maximum amount of epochs the model will train for.
            patience: Early-Stopping limit (epochs without improvement)
            before quitting a training process.
            tol_eps: Minimum absolute improvement for a model to be
            considered better than a previous checkpoint.

        Returns:
            Log of the training losses and metrics of the validation split
            of data.
        """
        track = dict()
        self.logger('Computing initial model performance...')
        assess_loader = val_loader if val_loader is not None else data_loader
        init_preds, current_loss = self.eval(assess_loader, verbose=True)
        self.best_loss = current_loss
        track['loss'] = [current_loss]

        if self.metrics:
            self.metrics.__init__()
            metrics_state = self.assess(assess_loader, init_preds)
            self.metrics.update(metrics_state)
            track = {**track, **{k: [val] for k, val in
                                     metrics_state.items()}}
        patience_left = patience
        for epoch in range(max_epochs):
            self.logger(f'Epoch {epoch + 1} / {max_epochs}')
            epoch_preds, epoch_loss = self.train_epoch(data_loader,
                                                   verbose=verbose)
            if val_loader is not None:
                epoch_preds, epoch_loss = self.eval(data_loader=val_loader)
            track['loss'].append(epoch_loss)
            metrics_state = self.assess(assess_loader, epoch_preds)
            if metrics_state is not None:
                for name, value in metrics_state.items():
                    track[name].append(value)
            if verbose:
                self.logger(f"Loss: {epoch_loss}; Metrics state"
                            f":{metrics_state}")

            if self.has_improved(epoch_loss, metrics_state, eps=tol_eps):
                patience_left = patience
                self.best_loss = epoch_loss
                if self.metrics:
                    self.metrics.update(metrics_state)
                self.logger('Saving new model...')
                self.best_model = copy.deepcopy(self.model)
            else:
                patience_left -= 1
                self.logger(f'Patience left: {patience_left} epochs.')
                if patience_left < 1:
                    self.logger('[Early-Stopping]: Leaving training loop...')
                    break
        return track

    def train_epoch(self, data_loader: Loader, verbose: bool = True
                    ) -> Tuple[Tensor, float]:
        """ A complete training pass throughout the samples of the dataset.

        Args:
            data_loader: Training&Validation data, must contain batches of
            inputs and labels.
            verbose: Whether to display progress bar or not.

        Returns:
            Average batch loss in the current epoch.
        """
        loss_sum = 0.0
        self.model.train()
        hide_bar = True if not verbose else False
        predictions = []
        for b, batch in enumerate(tqdm(data_loader, disable=hide_bar)):
            batch_preds, batch_loss = self._step(data=batch,
                                       step='train',
                                       use_best=False)
            predictions.append(batch_preds)
            loss_sum += batch_loss
        return torch.cat(predictions, axis=0), loss_sum / b

    def eval(self, data_loader: Loader, use_best: bool = False,
             verbose: bool = True) -> Tuple[Tensor, float]:
        """ Perform an evaluation on a dataset, comparing the predictions
        made by a learning model with the actual reference labels.

        Args:
            data_loader: Data to evaluate, must contain input features and
            labels.
            use_best: Whether to make predictions used saved best model.
            verbose: Whether to display progress bar or not.

        Returns:
            Predictions over the data considered.
            Average batch loss.
        """
        loss_sum = 0.0
        predictions = []
        self.model.eval() if not use_best else self.best_model.eval()
        hide_bar = True if not verbose else False
        with torch.no_grad():
            for b, batch in enumerate(tqdm(data_loader, disable=hide_bar)):
                batch_preds, batch_loss = self._step(data=batch,
                                                     step='eval',
                                                     use_best=use_best)
                predictions.append(batch_preds)
                loss_sum += batch_loss
        predictions = torch.cat(predictions, axis=0)
        return predictions, loss_sum / b

    def _step(self, data: Tuple, step: str = 'eval',
              use_best: bool = False) -> Tuple[Tensor, float]:
        """ Elaborate predictions over a batch of data.

        Args:
            data: Batch inputs and labels.
            step: Whether to perform training on weights or not.

        Returns:
            Batch predictions.
            Batch loss.
        """
        batch_inputs, batch_labels = data
        if isinstance(batch_inputs, list):
            batch_inputs = [
                {k: val.to(self.device) for k, val in x.items()}
                for x in batch_inputs
            ]
        else:
            batch_inputs = {k: val.to(self.device)
                            for k, val in batch_inputs.items()}
        batch_labels = batch_labels.to(self.device)
        if step != 'eval':
            self.optimizer.zero_grad()
        batch_preds = self.model(batch_inputs) if not use_best else \
            self.best_model(batch_inputs)
        batch_loss = self.loss_fn(
            target=torch.squeeze(batch_labels),
            input=torch.squeeze(batch_preds)
        )
        if step != 'eval':
            batch_loss.backward()
            self.optimizer.step()
        return batch_preds.detach(), batch_loss.detach().item()

    def assess(self, data_loader: Loader, predictions: Tensor) -> \
            Optional[Dict]:
        """ Compute the current value of a set of metrics.

        Args:
            data_loader: Dataset to evaluate on.
            predictions: (N,) predictions.

        Returns:
            Computed metrics if such a set is provided.
        """
        _labels = torch.cat(
            [batch[1].detach() for batch in data_loader], axis=0)
        if self.metrics:
            return self.metrics.compute(y=_labels, y_hat=predictions.cpu())
        return None

    def has_improved(self, current_loss: Tensor,
                     current_metrics: Dict = None, eps: float = 0.0) -> bool:
        """
        Check whether a model can be considered better than another one
        based on loss and/or metrics.

        Args:
            current_loss: Current loss value.
            current_metrics: Current value of a set of metrics.
            eps: Minimum improvement required.

        Returns:
            Whether a model's performance can be considered better.
        """
        if not current_metrics or self.monitor_metric == "loss":
            return self._has_improved_loss(current_loss, eps=eps)
        return self._has_improved_metrics(current_metrics, eps=eps)

    def _has_improved_loss(self, current_loss: Tensor, eps: float) -> bool:
        """ Whether loss has decreased.

        Args:
            current_loss: Current loss value

        Returns:
            Have we achieved a smaller loss value?
        """
        if self.best_loss < 0:
            threshold = self.best_loss + eps
            return True if current_loss > threshold else False
        threshold = self.best_loss - eps
        return True if current_loss < threshold else False

    def _has_improved_metrics(self, current_metrics: Dict, eps: float) -> bool:
        """ Whether the predictions made by a model outperform in terms of a
         set of metrics to that of previous steps.

        Args:
            current_metrics: Current value of a set of metrics.
            eps: Minimum percentual improvement required.

        Returns:
            Whether the metrics have seen an improvement or not.
        """
        mode = self.metrics[self.monitor_metric]['mode']
        current_value = current_metrics[self.monitor_metric]
        current_best = self.metrics[self.monitor_metric]
        if mode == "max":
            threshold = current_value - eps
            return True if threshold > current_best else False
        threshold = current_value + eps
        return True if threshold < current_best else False

    def save(self, to_file: Path, use_best: bool = True) -> None:
        """ Save model to file.

        Args:
            to_file: Path to save model in.
            use_best: If true, save self.best_model.

        """
        if use_best:
            torch.save(self.best_model.state_dict(), to_file)
        else:
            torch.save(self.model.state_dict(), to_file)
