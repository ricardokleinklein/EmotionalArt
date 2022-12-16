"""
Classification metrics base class spanning the most common metrics used for
model assessment, namely:
- Accuracy, Precision, Recall & F1-Score
- Jaccard index
- ROC-AUC for each category (TODO)
"""
import math
import numpy
import pandas
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from typing import Union, Dict, List, Optional, Tuple

from metrics.base_metrics import Metrics


Array = numpy.array
Series = pandas.Series
Tensor = torch.Tensor


def emd(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # https://discuss.pytorch.org/t/implementation-of-squared-earth-movers-distance-loss-function-for-ordinal-scale/107927
    return torch.mean(torch.mean(torch.square(torch.cumsum(y_true, dim=-1) -
                                              torch.cumsum(y_pred, dim=-1)),
                                 dim=-1))


class MultiLabelClsMetrics(Metrics):

    metrics = {
        'accuracy': {'fn': accuracy_score,
                     'value': 0,
                     'mode': 'max'},
        'precision': {'fn': precision_score,
                      'value': 0,
                      'mode': 'max'},
        'recall': {'fn': recall_score,
                   'value': 0,
                   'mode': 'max'},
        'f1': {'fn': f1_score,
               'value': 0,
               'mode': 'max'},
        'jaccard': {'fn': jaccard_score,
                    'value': 0,
                    'mode': 'max'}
    }

    def __init__(self, additional_metrics: Dict = None):
        """ Common metrics used to assess performance in multilabel,
        classification problems.

        Args:
            additional_metrics: Pairs of name: computing_function of other
        """
        super(MultiLabelClsMetrics, self).__init__(additional_metrics)

    def compute(self, y: Union[Tensor, Array, Series],
                y_hat: Union[Tensor, Array, Series]) -> Dict:
        """ Run through the metrics considered and compute their values
        given a set of reference values and a set of predictions.

        Args:
            y: (N,) Actual reference values.
            y_hat: (N,) Predicted values

        Returns:
            Value of the metrics considered.
        """
        metrics_ = dict()
        for key in self.metrics:
            if key == 'accuracy':
                metrics_[key] = self.metrics[key]['fn'](y, y_hat)
            elif key in ['precision', 'recall', 'f1_score', 'jaccard']:
                metrics_[key] = self.metrics[key]['fn'](
                    y, y_hat, average='weighted')
            elif key == 'ROC-AUC':
                raise FutureWarning('Feature to be implemented in future.')
        return metrics_


class DistributionDistanceMetrics(Metrics):

    metrics = {
        'acc@1': {'value': 0,
                  'mode': 'max'},
        'acc@3': {'value': 0,
                  'mode': 'max'},
        'acc@5': {'value': 0,
                  'mode': 'max'},
        'hamming': {'value': 0,
                    'mode': 'max'},
        'EMD': {'value': 0,
                'mode': 'min'}
    }

    def __init__(self, additional_metrics: Dict = None) -> Dict:
        super(DistributionDistanceMetrics, self).__init__(additional_metrics)
        self.top = [1, 3, 5]

    def compute(self, y: Tensor, y_hat: Tensor) -> Dict:
        """ Run through the metrics considered and compute their values
        given a set of reference values and a set of predictions.

        Args:
            y: (N, C) Actual label probability distribution.
            y_hat: (N, C) Predicted probability distribution

        Returns:
            Value of the metrics considered.
        """
        metrics_ = dict()
        metrics_['EMD'] = emd(y, y_hat).item()
        for k in self.top:
            y_k = torch.topk(y, k=k, dim=1)[1]
            y_hat_k = torch.topk(y_hat, k=k, dim=1)[1]


            y_k = y_k.detach().cpu().numpy()
            y_hat_k = y_hat_k.detach().cpu().numpy()
            if k > 1:
                metrics_['hamming'] = self.average_hamming(y_k, y_hat_k)
                metrics_[f"acc@{k}"] = self.accuracy(y_k, y_hat_k)
            else:
                metrics_[f'acc@{k}'] = self.accuracy(y_k, y_hat_k)
        return metrics_

    @staticmethod
    def average_hamming(y: numpy.ndarray, y_hat: numpy.ndarray) -> float:
        """ Compute the average hamming loss@3 for a series of samples
        independently.

        Args:
            y: (N, K) Target labels.
            y_hat: (N, K) Predicted labels.

        Returns:
            Average Hamming Loss
        """
        return numpy.mean([hamming_loss(i, j) for i, j in zip(y,
                                                              y_hat)]).item()

    @staticmethod
    def accuracy(y: numpy.ndarray, y_hat: numpy.ndarray) -> float:
        """ Compute accuracy scores for all the columns in a multilabel
        classification problem.

        Args:
            y: (N, K) Target labels.
            y_hat: (N, K) Predicted labels.

        Returns:
            Average accuracy score.
        """
        if y.shape[1] == 1:
            return accuracy_score(y.squeeze(), y_hat.squeeze())
        return numpy.mean([accuracy_score(i, j) for i, j in zip(y,
                                                                y_hat)]).item()
