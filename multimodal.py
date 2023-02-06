""" Multimodal

This script is devoted to the experimentation process using a multimodal
pipeline, fusing the different input in three ways:
    a) Computing the element-wise product
    b) Concatenating them
    c) Computing their vector product

The learning is always performed over Artemis data.

Positional arguments:
    src                     Artemis dataset file
    fusion                  Late-fusion method: [dot | mean | concat | normsum]
    embeddings_dir          Directory where precomputed embeddings are stored

Optional arguments:
    --val                   Whether to train according to val error
    -m, --monitor           Metric to assess the progress of the training
    -e, --epochs            Epochs to train for
    -p, --patience          Epochs to wait before early-stopping
    -b, --batch             Batch size
    --lr                    Initial learning rate
    --loss                  Loss function to use ["kldiv", "emd"]
    --eps                   Error tolerance for early-stopping
    --log-dir               Experiment dirname to label logs under
    -s, --save              Whether to save trained models afterwards
    --seed                  Random seed value
    --device               Device to use. Default: cuda
"""
import argparse
import numpy
import pandas
import torch.nn

from torch.nn import KLDivLoss
from data_preprocess.datasets import EmbeddingsDataset
from metrics.multilabel_classification import DistributionDistanceMetrics
from neural_models.linear import SimpleClassifier
from workflow.trainer import Trainer
from loggers.baselog import Logger


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    # Positional arguments
    parser.add_argument("src", type=str, help="Artemis dataset file")
    parser.add_argument("fusion", type=str,
                        choices=["dot", "concat", "mean", "normsum"],
                        help="Late-fusion method")
    parser.add_argument("embeddings_dir", type=str,
                        help="Directory where precomputed embeddings are "
                             "stored")

    # Optional arguments
    parser.add_argument("--val", action="store_true",
                        help="If set, evaluate over val data. test otherwise")
    parser.add_argument("-m", "--monitor", type=str, default="loss",
                        help="Metric to monitor during training")
    parser.add_argument("-e", "--epochs", type=int, default=500,
                        help="Number of epochs to train for")
    parser.add_argument("-p", "--patience", type=int, default=5,
                        help="Epochs before Early-Stopping")
    parser.add_argument("-b", "--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Initial learning rate")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Minimum improvement required during training")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Name for the experiment")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save models after training")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device on which to run the experiment")
    return parser.parse_args()


def tonumpy(str_dists: pandas.Series) -> numpy.ndarray:
    """

    """
    return numpy.array([
        numpy.fromstring(s[1:-1], dtype=float, sep=' ') for s in str_dists])


def main():
    args = parse_args()
    artemis = pandas.read_csv(args.src)
    if 'index' in list(artemis):
        artemis.set_index('index', inplace=True)
    loss = KLDivLoss(reduction="batchmean")
    metrics = DistributionDistanceMetrics()

    logger = Logger(log_dir=args.log_dir)
    logger(f"Experiment configuration:\n"
           f" {[f'{k}={v!r}' for k, v in args.__dict__.items()]}")

    if args.val:
        train_data = artemis[artemis['split'] == "train"]
        test_data = artemis[artemis['split'] == "val"]
    else:
        train_data = artemis[artemis['split'].isin(["train", "val"])]
        test_data = artemis[artemis['split'] == "test"]
    logger(f"\n\tNb train samples: {len(train_data)}\n\tNb eval samples: "
           f"{len(test_data)}")

    train = EmbeddingsDataset(data=train_data,
                              vector_dir=args.embeddings_dir,
                              scores=tonumpy(train_data["emotion"]),
                              method=args.fusion)
    train_loader = train.load(phase="train", batch_size=args.batch)

    test = EmbeddingsDataset(data=test_data,
                             vector_dir=args.embeddings_dir,
                             scores=tonumpy(test_data["emotion"]),
                             method=args.fusion)
    test_loader = test.load(phase="test", batch_size=args.batch)
    val_loader = test_loader if args.val else None

    nb_emotions = len(artemis['emotion_label'].unique())
    input_features = len(train[0][0]['vector'])
    model = SimpleClassifier(in_features=input_features,
                             num_classes=nb_emotions)
    logger(f"\n\tNb categories: {nb_emotions}")

    trainer = Trainer(model=model, loss_fn=loss,
                      metrics=metrics, monitor_metric=args.monitor,
                      device=args.device, learning_rate=args.lr, logger=logger)

    trainer.fit(data_loader=train_loader, val_loader=val_loader,
                max_epochs=args.epochs, patience=args.patience,
                tol_eps=args.eps)

    test_preds, test_loss = trainer.eval(data_loader=test_loader,
                                         use_best=True, verbose=True)
    test_metrics = trainer.assess(data_loader=test_loader,
                                  predictions=test_preds)
    logger(f"Loss: {test_loss}\nMetrics:{test_metrics}")


if __name__ == "__main__":
    main()
