"""
Base class for the implementation of the functionality of a set of metrics.
"""
import numpy
import pandas
import torch

from typing import Any, Dict, Union, Callable


Array = numpy.array
Series = pandas.Series
Tensor = torch.Tensor


class Metrics:

    metrics: Dict[str, Dict[str, Union[Callable, str, int]]] = None

    def __init__(self, additional_metrics: Dict = None):
        """ Base class for the computing and tracking of metrics of interest
         during an experiment. Each such metric is implemented as a key in a
         dictionary whose values are in turn dictionaries with the necessary
         information.

        Attributes:
            metrics: A dictionary of metrics of interest with the fields:
                fn: Pairs of metric names & functions to compute them.
                value: Metric names & current values of interest (best).
                mode: Whether max or min implies a better performance on a
                metric.

        Args:
            additional_metrics: Further metrics to compute, in the same
            formatting as ``metrics``.

        """
        if additional_metrics:
            self.metrics = {**self.metrics, **additional_metrics}

    def compute(self, y: Union[Tensor, Array, Series],
                y_hat: Union[Tensor, Array, Series]) -> Dict:
        """ Run throughout the metrics to be considered and compute their
        updated values according to the inputs and targets presented.

        Args:
            y: (N,) Actual reference values.
            y_hat: (N,) Predicted values.

        Returns:
            Metrics' value.
        """
        return {key: self.metrics[key]['fn'](y, y_hat) for key in self.metrics}

    def update(self, new_metric_values: Dict) -> None:
        """ Set new metric values as default.

        Args:
            new_metric_values: New values for each metric in the metrics set.
        """
        for metric_name in new_metric_values:
            self.metrics[metric_name]['value'] = new_metric_values[metric_name]

    def __getitem__(self, metric_name) -> Union[int, float]:
        """ Retrieve the current value of a metric.

        Args:
            metric_name: Which metric to retrieve

        Returns:
            Value of a metric. Can be any number.
        """
        return self.metrics[metric_name]['value']