""" Baseline

Experiments performed to assess our baseline based on the different branches
of a pretrained CLIP model.

Positional arguments:
    src                     Dataset file.

Optional arguments:
    -m, --monitor           Metric to assess the progress of the training
    -p, --patience          Epochs to wait before early-stopping
    -b, --batch             Batch size
    --name                  Name of the experiment
    --save_models           Whether to save models after each apoch

"""
import argparse
import numpy
import pandas
from torch.nn import KLDivLoss
from data_preprocess.tokenizers import BPETokenizer
from data_preprocess.datasets import SentencesDataset
from neural_models.transformers import CustomTextualCLIP
from metrics.multilabel_classification import DistributionDistanceMetrics
from metrics.multilabel_classification import emd
from workflow.kfolds import KFoldExperiment


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original Artemis CSV")

    parser.add_argument("-f", "--finetune", action="store_true",
                        help="If True, train only classification layer(s)")
    parser.add_argument("-m", "--monitor", type=str, default="loss",
                        help="Metric to monitor during training")
    parser.add_argument("-e", "--epochs", type=int, default=500,
                        help="Number of epochs to train for")
    parser.add_argument("-p", "--patience", type=int, default=5,
                        help="Epochs before Early-Stopping")
    parser.add_argument("-b", "--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--loss", type=str, default='kldiv',
                        choices=['kldiv', 'emd'], help="Loss function")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Minimum improvement required during training")
    parser.add_argument("-k", "--kfolds", type=int, default=5,
                        help="Number of folds")
    parser.add_argument("--name", type=str, default="KfoldExp",
                        help="Name for the experiment")
    parser.add_argument("-s", "--save", type=str,
                        help="Save models after training")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    artemis = pandas.read_csv(args.src)
    loss = KLDivLoss(reduction='batchmean') if args.loss == "kldiv" else emd
    metrics = DistributionDistanceMetrics()
    tokenizer = BPETokenizer('clip', seq_len=77)

    experiment = KFoldExperiment(data_reader=SentencesDataset,
                                 num_folds=args.kfolds,
                                 max_epochs=args.epochs,
                                 metrics=metrics,
                                 monitor_metric=args.monitor,
                                 patience=args.patience,
                                 random_seed=args.seed,
                                 name=args.name,
                                 save_models=args.save)
    ground_truth = numpy.array([
        numpy.fromstring(s[1:-1], dtype=float, sep=' ') for s in \
        artemis["emotion"]]
    )
    model = CustomTextualCLIP(num_classes=ground_truth.shape[1],
                              finetune=args.finetune, multisentence=True)
    results_logging = experiment(X=artemis['utterance'].values,
                                 target=ground_truth,
                                 processor=tokenizer,
                                 model=model,
                                 loss_fn=loss,
                                 batch_size=args.batch)
    print(results_logging)


if __name__ == "__main__":
    main()
