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
from sklearn.metrics import roc_auc_score
from typing import Union, Dict

from base_metrics import Metrics


Array = numpy.array
Series = pandas.Series
Tensor = torch.Tensor


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
