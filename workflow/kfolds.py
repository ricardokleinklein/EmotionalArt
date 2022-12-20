"""
K-Fold Cross-Validation tools and auxiliary functions
"""
import numpy
import pandas
import random
import torch
import torch.nn as nn
import transformers
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from sklearn.model_selection import KFold
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torch import device

from metrics.base_metrics import Metrics
from workflow.trainer import Trainer


def set_device(device: Optional[str] = None) -> device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if device is not None:
        return torch.device(device)
    nb_gpu = torch.cuda.device_count()
    if nb_gpu == 1:
        return torch.device("cuda")
    for gpu in range(nb_gpu):
        if torch.cuda.memory_usage(gpu) == 0:
            return torch.device(f"cuda:{gpu}")


def make_splits(train_idxs: List, test_idxs: List,
                X: Union[numpy.ndarray, pandas.Series],
                target: Union[numpy.ndarray, pandas.Series],
                ) ->  Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    """Arranges data in train, validation and test splits.

    Args:
        train_idxs: Indices of samples aimed at training.
        test_idxs: Indices of samples aimed at testing.
        X: Input features.
        target: Target labels.

    Returns:
        a tuple of (input features, labels) for the train/test splits.
    """
    X_train, X_test = X[train_idxs], X[test_idxs]
    y_train, y_test = target[train_idxs], target[test_idxs]
    return (X_train, y_train), (X_test, y_test)


class KFoldExperiment:

    def __init__(self, data_reader: torch.utils.data.Dataset,
                 num_folds: int = 5,
                 max_epochs: int = 500,
                 patience: int = 5,
                 metrics: Metrics = None,
                 monitor_metric: str = 'loss',
                 random_seed: int = 1234,
                 log_dir: Optional[Union[str, Path]] = None,
                 save_models: bool = False,
                 device: str = "cuda") -> None:
        """ K-Fold Cross Validation Experimentation pipeline.

        Args:
            data_reader: How to process experiment data.
            num_folds: Number of folds to arrange data within.
            max_epochs: Maximum amount of epochs the model will train for.
            patience: Early-Stopping limit (epochs without improvement)
            before quitting a training process.
            metrics: Relevant metrics to take into account.
            monitor_metric: Name of the metric to assess a model by.
            random_seed: Initial random seed.
            log_dir: Name assigned to the experiment.
            save_models: If True, save models.
            device: Device on which to run the experiment.
        """
        self.data_reader = data_reader
        self.num_folds = num_folds
        self.max_epochs = max_epochs
        self.patience = patience
        self.metrics = metrics
        self.monitor_metric = monitor_metric
        self.seed = random_seed
        self.name = log_dir
        self.save_models = save_models

        self._set_random_seed()
        self.device = set_device(device=device)

        self.k_folder = KFold(n_splits=num_folds, shuffle=True,
                              random_state=random_seed)

    def _set_random_seed(self):
        """ Fix the initial random seed for the experiment."""
        torch.backends.cudnn.deterministic = True
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        numpy.random.seed(self.seed)

    def __call__(self, X: Union[numpy.ndarray, pandas.Series],
                 target: Union[numpy.ndarray, pandas.Series],
                 processor: transformers.PreTrainedTokenizer,
                 model: nn.Module,
                 loss_fn: Union[nn.Module, Callable],
                 batch_size: int = 32,
                 eps: float = 0.05,
                 learning_rate: float = 1e-5,
                 logger: Optional[SummaryWriter] = None
                 ) -> Dict:
        """ Proceed with the experimentation over the folds, each as a
        separate trial.

        Args:
            X: Input features.
            target: Target labels.
            model: Neural model.
            batch_size: Batch size.
            learning_rate: Initial learning rate.

        Returns:
            Fold-wise summary of validation and test results.
        """
        fold_generator = self.k_folder.split(X, target)
        fold_results = dict()
        for k, fold_k in enumerate(fold_generator):
            fold_str = f'\n[NEW FOLD: {k+1}/{self.num_folds}]'
            print(fold_str)
            train_idxs, test_idxs = fold_k
            train, test = make_splits(train_idxs, test_idxs, X, target)

            if logger is not None:
                suffix = f"fold_{k+1}"
                logger.filename_suffix = suffix
                logger.write(f"{fold_str}\n\t#Train samples: "
                             f"{len(train[1])}\n\t# Test samples: "
                             f"{len(test[1])}")

            train_loader = self.data_reader(train[0], train[1],
                processor=processor).load('train', batch_size)
            test_loader = self.data_reader(test[0], test[1],
                processor=processor).load('test', batch_size)

            trainer = Trainer(model=model,
                              loss_fn=loss_fn,
                              metrics=self.metrics,
                              monitor_metric=self.monitor_metric,
                              device=self.device,
                              learning_rate=learning_rate,
                              logger=logger,
                              verbose=True)
            trainer.fit(data_loader=train_loader,
                        max_epochs=self.max_epochs, patience=self.patience,
                        tol_eps=eps)

            test_preds, test_loss = trainer.eval(data_loader=test_loader,
                                                 use_best=True,
                                                 verbose=True)
            test_metrics = trainer.assess(data_loader=test_loader,
                                          predictions=test_preds)
            fold_results[f'fold_{k+1}'] = test_metrics
            print(f'[TEST IN FOLD {k+1}/{self.num_folds}')
            print(f"Loss: {test_loss.item()}\nMetrics:{test_metrics}")

            if self.save_models is not None:
                trainer.save(self.save_models / (self.name + f"_fold_{k}.pt"))

        return fold_results
