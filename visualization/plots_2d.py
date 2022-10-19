"""
Visualization tools in 2 dimensions.
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy
import torch

from typing import List, Tuple, Union, Optional


Tensor = torch.Tensor


def scored_scatter(
        X: Union[List[numpy.ndarray], numpy.ndarray],
        scores: Union[list, numpy.ndarray],
        score_name: str,
        point_size: int = 25
)-> None:
    """
    Plot a scatter of points whose color is determined by a score label.

    Args:
        X (Union[List[numpy.ndarray], numpy.ndarray]): (N, 2) Input data.
        scores (Union[list, numpy.ndarray]): (N,) score labels.
        score_name (str): Name of the score used.
        point_size (int): Point size in the plot.
    """
    mu = numpy.mean(scores)
    lower = min(scores)
    upper = max(scores)
    fig, ax = plt.subplots(figsize=(20, 15))
    norm = colors.TwoSlopeNorm(
        vmin=lower,
        vmax=upper,
        vcenter=mu
    )
    scatter = ax.scatter(X[:, 0], X[:, 1], s=point_size, c=scores,
                         cmap=cm.get_cmap('RdYlGn'), norm=norm,
                         edgecolor='k', linewidth=0.1)
    plt.axis('off')
    cbar = fig.colorbar(scatter)
    cbar.ax.get_yaxis().labelpad = 35
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_ylabel(score_name, rotation=270, fontsize=32)
    plt.tight_layout()
    plt.show()


def cluster_scatter(
        X: Union[List[numpy.ndarray], numpy.ndarray],
        labels: Union[list, numpy.ndarray],
        label_names: Union[list, numpy.ndarray],
        include_noise: bool = True,
        point_size: int = 25
) -> None:
    """
    Draw a scatter plot in which color comes determined by the sample's label.

    Args:
        X (Union[List[numpy.ndarray], numpy.ndarray]): (N, 2)Data points.
        labels (Union[list, numpy.ndarray]): (N,) Sample labels.
        label_names (Union[list, numpy.ndarray]): (N_labels,) Names of classes.
        include_noise (bool): Plot noise samples (label = -1).
        point_size (int): Point size in the plot
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    if include_noise:
        outliers = X[numpy.where(labels == -1)[0]]
        plt.scatter(outliers[:, 0], outliers[:, 1], color='#BDBDBD',
                    s=point_size//2)
        not_noise = numpy.where(labels != -1)[0]
        X = X[not_noise]
        labels = labels[not_noise]

    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=point_size,
                          cmap=cm.get_cmap('jet', max(labels) + 1),
                          edgecolor='k', linewidth=0.1)
    cbar = fig.colorbar(scatter, ticks=numpy.linspace(0 + 0.5, max(labels) -
                                                      0.5, max(labels) + 1))
    if label_names:
        cbar.ax.set_yticklabels(label_names)
    cbar.ax.tick_params(labelsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def boxplot(distributions: numpy.ndarray, labels: list = None) -> None:
    """
    Draws a boxplot histogram for a set of distributions.

    Args:
        distributions (numpy.ndarray): (D, N) D distributions with sampling N.
        labels (list): Name assigned to each distribution.
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    bplot = ax.boxplot(distributions,
                       vert=False,
                       labels=labels,
                       patch_artist=True)
    if labels:
        labels = [', '.join(label) for label in labels]
        cmap = plt.cm.get_cmap('jet', len(labels) + 1)
        for patch, color in zip(bplot['boxes'], range(len(labels))):
            patch.set_facecolor(cmap(color))
    plt.grid()
    ax.tick_params(labelsize=22)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    plt.tight_layout()
    plt.show()


def display_attentions(attention_layer: Tensor,
                       n_rows : int,
                       n_cols: int) -> None:
    """ Display an overview of the attention head values.

    Args:
        attention_layer: Attention weight values.
        n_rows: Number of rows.
        n_cols: Number of columns.

    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
    for h, head in enumerate(attention_layer):
        row, col = h // (n_rows +1), h % n_cols
        axes[row, col].matshow(head.detach().numpy(), cmap='gray')
        axes[row, col].set_title(f'Head #{h}')
    fig.tight_layout()


def display_attention_head(head: Tensor, tokens: List[str]):
    """ Display the matrix of weights of a given attention head.

    Args:
        head: Weight values.
        tokens: Components the attention sequence is referred to.

    """
    plt.matshow(head.detach().numpy(), cmap='gray')
    plt.xticks(ticks=list(range(len(tokens))),
               labels=tokens, rotation='vertical')
    plt.yticks(ticks=list(range(len(tokens))),
               labels=tokens, rotation='horizontal')
    plt.tight_layout()
