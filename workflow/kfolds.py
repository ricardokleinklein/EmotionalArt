"""
K-Fold Cross-Validation tools and auxiliary functions
"""
import numpy
import pandas
import random
import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from metrics.base_metrics import Metrics
from workflow.trainer import Trainer

Array = numpy.array
Series = pandas.Series
Dataset = torch.utils.data.Dataset
Loader = torch.utils.data.DataLoader


def detect_device():
    """ Is there any GPU available? If so, which one to use?"""
    # TODO: Manually select which GPU you wish to use as an arg.
    # TODO: Verbosa info about found GPUs.
    if not torch.cuda.is_available():
        return "cpu"
    # torch.cuda.empty_cache()
    return "cuda:0" if torch.cuda.device_count() > 1 else "cuda"


def make_splits(train_idxs: List, test_idxs: List, X: Union[Array, Series],
                 target: Union[Array, Series], val_size: float = 0.2) ->  \
        Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    """Arranges data in train, validation and test splits.

    Args:
        train_idxs: Indices of samples aimed at training.
        test_idxs: Indices of samples aimed at testing.
        X: Input features.
        target: Target labels.
        val_size: Percentage of the training data to use for validation
        purposes.

    Returns:
        a tuple of (input features, labels) for the train/val/test splits.
    """
    X_train_, X_test = X[train_idxs], X[test_idxs]
    y_train_, y_test = target[train_idxs], target[test_idxs]
    X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_,
                                                      test_size=val_size,
                                                      random_state=1234,
                                                      shuffle=True)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class KFoldExperiment:

    def __init__(self, data_reader: Dataset,
                 num_folds: int = 5,
                 max_epochs: int = 500,
                 patience: int = 5,
                 metrics: Metrics = None,
                 monitor_metric: str = 'loss',
                 random_seed: int = 1234,
                 device: Optional[str] = None,
                 **kwargs):
        """

        Args:
            data_reader: How to process experiment data.
            num_folds: Number of folds to arrange data within.
            max_epochs: Maximum amount of epochs the model will train for.
            patience: Early-Stopping limit (epochs without improvement)
            before quitting a training process.
            metrics: Relevant metrics to take into account.
            monitor_metric: Name of the metric to assess a model by.
            random_seed: Initial random seed.
            device: Processor (GPU or CPU).
        """
        self.data_reader = data_reader
        self.num_folds = num_folds
        self.max_epochs = max_epochs
        self.patience = patience
        self.metrics = metrics
        self.monitor_metric = monitor_metric
        self.seed = random_seed

        self._set_random_seed()
        device_name = device if device is not None else detect_device()
        self.device = torch.device(device_name)

        self.kwargs = kwargs

    def _set_random_seed(self):
        """ Fix the initial random seed for the experiment."""
        torch.backends.cudnn.deterministic = True
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        numpy.random.seed(self.seed)

    def run(self, X: Union[Array, Series],
            target: Union[Array, Series],
            model: nn.Module,
            loss_fn: Union[nn.Module, Callable],
            batch_size: int = 32) -> Dict:
        """ Proceed with the experimentation over the folds, each as a
        separate trial.

        Args:
            X: Input features.
            target: Target labels.
            model: Neural model.
            batch_size: Batch size.

        Returns:
            Fold-wise summary of validation and test results.
        """
        kfolds = KFold(n_splits=self.num_folds, shuffle=True,
                       random_state=self.seed)
        fold_generator = kfolds.split(X, target)
        fold_results = dict()
        for k, fold_k in enumerate(fold_generator):
            print(f'[NEW FOLD: {k+1}/{self.num_folds}]')
            train_idxs, test_idxs = fold_k
            train, val, test = make_splits(train_idxs, test_idxs, X, target)

            train_loader = self.data_reader(
                train[0].values,
                train[1].values,
                **self.kwargs).load('train', batch_size)
            val_loader = self.data_reader(
                val[0].values,
                val[1].values,
                **self.kwargs).load('val', batch_size)
            test_loader = self.data_reader(
                test[0].values,
                test[1].values,
                **self.kwargs).load('test', batch_size)

            trainer = Trainer(model=model,
                              loss_fn=loss_fn,
                              metrics=self.metrics,
                              monitor_metric=self.monitor_metric,
                              device=self.device,
                              verbose=False)

            val_log = trainer.fit(train_data_loader=train_loader,
                                  val_data_loader=val_loader,
                                  max_epochs=self.max_epochs,
                                  patience=self.patience)
            test_preds, test_loss = trainer.eval(data_loader=test_loader,
                                                 use_best=True,
                                                 verbose=True)
            test_metrics = trainer.assess(data_loader=test_loader,
                                          predictions=test_preds)
            fold_results[f'fold_{k+1}'] = test_metrics
        return fold_results