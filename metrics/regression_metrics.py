"""
Regression metrics base class spanning the most common metrics used for
model assessment, namely:
- Mean Square Error
- Pearson's correlation coefficient.
- Spearman's rank correlation coefficient.
- Explained variance ratio.
- Max error between prediction and reference.
"""
import math
import numpy
import pandas
import torch
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from typing import Union, Dict

from metrics.base_metrics import Metrics


Array = numpy.array
Series = pandas.Series
Tensor = torch.Tensor


def calc_pearson(y: Union[Tensor, Array, Series],
                 y_hat: Union[Tensor, Array, Series]) -> float:
    """ Pearson's Correlation coefficient between two sets.

    Args:
        y: (N,) Actual reference values.
        y_hat: (N,) Predicted values

    Returns:
        Index's value between [-1, 1].
    """
    score, p_value = pearsonr(y, y_hat)
    if math.isnan(score):
        return 0
    else:
        return score


def calc_spearman(y: Union[Tensor, Array, Series],
                 y_hat: Union[Tensor, Array, Series]) -> float:
    """ Spearman's Rank Correlation coefficient between two sets.

    Args:
        y: (N,) Actual reference values.
        y_hat: (N,) Predicted values

    Returns:
        Index's value between [-1, 1].
    """
    score, p_value = spearmanr(y, y_hat)
    if math.isnan(score):
        return 0
    else:
        return score


class RegressionMetrics(Metrics):

    metrics = {
        'MSE': {'fn': mean_squared_error,
                'value': 0,
                'mode': 'min'},
        'Pearson': {'fn': calc_pearson,
                    'value': -1,
                    'mode': 'max'},
        'Spearman': {'fn': calc_spearman,
                     'value': -1,
                     'mode': 'max'},
        'ExplainedVariance': {'fn': explained_variance_score,
                              'value': 0,
                              'mode': 'max'},
        'MaxError': {'fn': max_error,
                     'value': 1,
                     'mode': 'min'}
    }

    def __init__(self, additional_metrics: Dict = None):
        """ Common metrics used in regression problems."""
        super(RegressionMetrics, self).__init__(additional_metrics)
