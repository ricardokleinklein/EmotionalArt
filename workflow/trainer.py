"""
Main training tools, wrapped around a single Trainer object.

The Trainer interfase allows its usage either independently or within the
context of a bigger experimentation pipeline. It implements both the
training and testing utilities, so it doesn't matter whether you're just
looking for a system to evaluate your pretrained model, go through the
Trainer class for a smooth experience.

TODO: Swap to a cleaner LOG system.
TODO: Clean up verbose.
TODO: Save best model in disk.
TODO: Enable changing the optimizer.
TODO: Enable args-based hyperparameter tuning.
"""
import copy
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Callable, Dict, Optional, Union, Tuple

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
                 *args, **kwargs):
        """ Systematize a typical lifecycle of a machine learning
        pipeline, including training, validating and testing one.

        Args:
            model: Torch model to work with.
            loss_fn: Error function.
            metrics: Metrics to assess the training and evaluation by.
            monitor_metric: Monitor to ponder performance.
            device: Which device to use (CPU / GPU).
        """
        self.model = copy.deepcopy(model).to(device)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.monitor_metric = monitor_metric
        self.device = device

        self.best_model = copy.deepcopy(model)
        self.best_loss = 0
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=kwargs.get('lr', 1e-5))

    def fit(self, train_data_loader: Loader, val_data_loader: Loader,
            max_epochs: int = 500, patience: int = 5, verbose: bool = True) \
            -> Dict:
        """ Automatic pipeline for training a model.

        Args:
            train_data_loader: Training data, ust contain inputs and labels.
            val_data_loader: Data to evaluate, must contain input features and
            labels.
            max_epochs: Maximum amount of epochs the model will train for.
            patience: Early-Stopping limit (epochs without improvement)
            before quitting a training process.
            verbose:

        Returns:
            Log of the training losses and metrics of the validation split
            of data.
        """
        val_track = dict()
        print('Computing initial model performance...')
        init_preds, current_loss = self.eval(val_data_loader, verbose=True)
        self.best_loss = current_loss
        val_track['loss'] = [float(current_loss.cpu().detach().numpy())]

        if self.metrics:
            self.metrics.__init__()
            val_metrics_state = self.assess(val_data_loader, init_preds)
            self.metrics.update(val_metrics_state)
            val_track = {**val_track, **{k: [val] for k, val in
                                         val_metrics_state.items()}}
        patience_left = patience
        for epoch in range(max_epochs):
            print(f'Epoch {epoch + 1} / {max_epochs}')
            _ = self.train_epoch(train_data_loader, verbose=verbose)
            val_preds, val_loss = self.eval(val_data_loader, verbose=verbose)
            val_track['loss'].append(float(val_loss.cpu().detach().numpy()))
            val_metrics_state = self.assess(val_data_loader, val_preds)
            print(val_metrics_state)
            if val_metrics_state is not None:
                for name, value in val_metrics_state.items():
                    val_track[name].append(value)

            if self.has_improved(val_loss, val_metrics_state):
                patience_left = patience
                self.best_loss = val_loss
                if self.metrics:
                    self.metrics.update(val_metrics_state)
                print('Saving new model...')
                self.best_model = copy.deepcopy(self.model)
            else:
                patience_left -= 1
                print(f'Patience left: {patience_left} epochs.')
                if patience_left < 1:
                    print('[Early-Stopping]: Leaving training loop...')
                    break
        return val_track

    def train_epoch(self, data_loader: Loader, verbose: bool = True) -> float:
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
        for b, batch in enumerate(tqdm(data_loader, disable=hide_bar)):
            _, batch_loss = self._step(data=batch,
                                       step='train',
                                       use_best=False)
            loss_sum += batch_loss
        return loss_sum / b

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
        batch_inputs = [
            {k: val.to(self.device) for k, val in x.items()}
            for x in batch_inputs
        ]
        batch_labels = batch_labels.to(self.device)
        if step != 'eval':
            self.optimizer.zero_grad()
        batch_preds = self.model(batch_inputs) if not use_best else \
            self.best_model(batch_inputs)
        batch_loss = self.loss_fn(
            torch.squeeze(batch_labels),
            torch.squeeze(batch_preds)
        )
        if step != 'eval':
            batch_loss.backward()
            self.optimizer.step()
        return batch_preds, batch_loss

    def assess(self, data_loader: Loader, predictions: Tensor) -> \
            Optional[Dict]:
        """ Compute the current value of a set of metrics.

        Args:
            data_loader: Dataset to evaluate on.
            predictions: (N,) predictions.

        Returns:
            Computed metrics if such a set is provided.
        """
        _labels = torch.cat([batch[1] for batch in data_loader], axis=0)
        if self.metrics:
            return self.metrics.compute(y=_labels, y_hat=predictions)
        return None

    def has_improved(self, current_loss: Tensor,
                     current_metrics: Dict = None) -> bool:
        """
        Check whether a model can be considered better than another one
        based on loss and/or metrics.

        Args:
            current_loss: Current loss value.
            current_metrics: Current value of a set of metrics.

        Returns:
            Whether a model's performance can be considered better.
        """
        if not current_metrics or self.monitor_metric == "loss":
            return self._has_improved_loss(current_loss)
        return self._has_improved_metrics(current_metrics)

    def _has_improved_loss(self, current_loss: Tensor) -> bool:
        """ Whether loss has decreased.

        TODO: Extend to losses that have to increase.

        Args:
            current_loss: Current loss value

        Returns:
            Have we achieved a smaller loss value?
        """
        return True if current_loss < self.best_loss else False

    def _has_improved_metrics(self, current_metrics: Dict) -> bool:
        """ Whether the predictions made by a model outperform in terms of a
         set of metrics to that of previous steps.

        Args:
            current_metrics: Current value of a set of metrics.

        Returns:
            Whether the metrics have seen an improvement or not.
        """
        mode = self.metrics[self.monitor_metric]['mode']
        current_value = current_metrics[self.monitor_metric]
        current_best = self.metrics[self.monitor_metric]
        if mode == "max":
            return True if current_value > current_best else False
        return True if current_value < current_best else False
