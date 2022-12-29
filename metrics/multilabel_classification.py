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


def emd(target: torch.Tensor, input: torch.Tensor,
        log_space: bool = True) -> torch.Tensor:
    """ Earth-Moving Distance function.

    Args:
        target: Reference distribution.
        input: Predicted distribution
        log_space: If True, predictions come in the log space.

    Returns:
        EMD between both distribution
    """
    if log_space:
        input = torch.exp(input)
    # https://discuss.pytorch.org/t/implementation-of-squared-earth-movers-distance-loss-function-for-ordinal-scale/107927
    return torch.mean(torch.mean(torch.square(torch.cumsum(target, dim=-1) -
                                              torch.cumsum(input, dim=-1)),
                                 dim=-1))


class BatchWiseClsMetrics(Metrics):
    metrics = {
        'text_accuracy': {'fn': accuracy_score,
                     'value': 0,
                     'mode': 'max'},
        'vision_accuracy': {'fn': accuracy_score,
                     'value': 0,
                     'mode': 'max'}
    }

    def __init__(self, additional_metrics: Dict = None) -> None:
        """ Common metrics used to assess performance in multilabel,
        classification problems when the predictions can only be understood
        at a batch level. For instance, because the labels are relative to
        batch position.

        Note: Currently implemented only for 2-modal cases. Further cases
        should modify the compute method.

        Args:
            additional_metrics: Pairs of name: computing_function of other
        """
        super(BatchWiseClsMetrics, self).__init__(additional_metrics)

    def compute(self, y: Union[torch.Tensor, Array, Series],
                y_hat: Union[torch.Tensor, Array, Series]) -> Dict:
        """Run through the metrics considered and compute their values
        given a set of reference values and a set of predictions.

        Args:
            y: Not used.
            y_hat: (NB_BATCHS, BS, BS) Predicted values per batch

        Returns:
            Value of the metrics considered.
        """
        size = y_hat.shape[-1]
        gt = torch.arange(size).cpu().numpy()
        metrics_ = dict()
        for batch in y_hat:
            preds_vision = torch.argmax(batch[1], dim=0).cpu().numpy()
            preds_text = torch.argmax(batch[1], dim=0).cpu().numpy()
            metrics_["vision_accuracy"] = accuracy_score(gt, preds_vision)
            metrics_["text_accuracy"] = accuracy_score(gt, preds_text)
        return metrics_


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
        'hamming@3': {'value': 0,
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
        y1 = torch.topk(y, k=1, dim=1)[1].detach().cpu().numpy()
        y_hat1 = torch.topk(y_hat, k=1, dim=1)[1].detach().cpu().numpy()
        metrics_['acc@1'] = accuracy_score(y1.squeeze(), y_hat1.squeeze())

        y3 = torch.topk(y, k=3, dim=1)[1].detach().cpu().numpy()
        y_hat3 = torch.topk(y_hat, k=3, dim=1)[1].detach().cpu().numpy()
        metrics_['hamming@3'] = self.average_hamming(y3, y_hat3)
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

