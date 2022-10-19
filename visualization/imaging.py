"""
Visualization of images and similar sources of data.
"""

import numpy
import librosa.display as display
import matplotlib.pyplot as plt

from typing import Optional


Array = numpy.array


def plot_spectrogram(spectrogram: Array, sampling_rate: float,
                     hop_length: int = 512, y_axis: str = "log",
                     save_to: Optional[str] = None):
    """ Plot an spectrogram, in either log or linear scale on the Y-axis.

    Args:
        spectrogram: Spectrogram to plot.
        sampling_rate: Frequency at which audio is sampled.
        hop_length: Hop size in windowing.
        y_axis: Whether to set the Y-axis on a linear or log scale.
        save_to: File path to save plot as a normal PNG image.

    Returns:
        None
    """
    fig = plt.figure(num=1, clear=True, figsize=(7, 5))
    fig.add_subplot()
    display.specshow(spectrogram, sr=sampling_rate, hop_length=hop_length,
                     x_axis="Time (s)", y_axis=y_axis)
    plt.tight_layout()
    if save_to:
        plt.axis('off')
        plt.savefig(save_to)
    else:
        plt.colorbar(format="%+2.f")
