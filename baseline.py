""" Baseline

Experiments performed to assess our baseline based on the different branches
of a pretrained CLIP model.

Positional arguments:
    src                     Dataset file.

Optional arguments:
    --val                   Whether to train according to val error
    -f, --finetune          If set, train only last layer
    -m, --monitor           Metric to assess the progress of the training
    -e, --epochs            Epochs to train for
    -p, --patience          Epochs to wait before early-stopping
    -b, --batch             Batch size
    --lr                    Initial learning rate
    --loss                  Loss function to use ["kldiv", "emd"]
    --eps                   Error tolerance for early-stopping
    --log-dir               Experiment dirname to label logs under
    -s, --save              Whether to save trained models afterwards
    --export                Export test predictions to file
    --seed                  Random seed value
    --device               Device to use. Default: cuda

"""
import argparse
import numpy
import pandas
from torch.nn import KLDivLoss, BCELoss
from data_preprocess.tokenizers import BPETokenizer
from data_preprocess.datasets import SentencesDataset, ImageDataset
from neural_models.transformers import CustomTextualCLIP, CustomVisualCLIP
from metrics.multilabel_classification import DistributionDistanceMetrics
from metrics.multilabel_classification import emd
from workflow.trainer import Trainer
from loggers.baselog import Logger


BRANCH_CONFIG = {'text': {'reader': SentencesDataset,
                          'feat_col': 'utterance',
                          'processor': BPETokenizer('clip', seq_len=77)},
                 'vision': {'reader': ImageDataset,
                            'feat_col': 'localpath',
                            'processor': None},
                 'late': None,
                 'align': None}


LOSS = {'kldiv': KLDivLoss(reduction='batchmean'),
        'emd': emd,
        'entropy': BCELoss(reduction='mean')}


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original Artemis CSV")
    parser.add_argument("branch", type=str, default="vision",
                        choices=["text", "vision", "late", "align"],
                        help="Branch to experiment with, or fusion style")
    parser.add_argument("--val", action="store_true",
                        help="If set, evaluate over val data. test otherwise")
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
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Initial learning rate")
    parser.add_argument("-d", "--decay-lr", action="store_true",
                        help="Implement learning rate decay")
    parser.add_argument("--loss", type=str, default='kldiv',
                        choices=['kldiv', 'emd', 'entropy'],
                        help="Loss function")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Minimum improvement required during training")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Name for the experiment")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save models after training")
    parser.add_argument("--export", action="store_true",
                        help="Export test predictions to file")
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
    # Read command line arguments
    args = parse_args()
    artemis = pandas.read_csv(args.src)
    loss = LOSS[args.loss]

    # Experiment environment: metrics, logger, ground-truth...
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
    data_reader = BRANCH_CONFIG[args.branch]['reader']
    feat_col = BRANCH_CONFIG[args.branch]['feat_col']
    processor = BRANCH_CONFIG[args.branch]['processor']

    train = data_reader(train_data[feat_col].values,
                        scores=tonumpy(train_data['emotion']),
                        processor=processor)
    train_loader = train.load(phase="train", batch_size=args.batch)
    test = data_reader(test_data[feat_col].values,
                       scores=tonumpy(test_data['emotion']),
                       processor=processor)
    test_loader = test.load(phase="test", batch_size=args.batch)
    val_loader = test_loader if args.val else None

    nb_emotions = len(artemis['emotion_label'].unique())
    logger(f"\n\tNb categories: {nb_emotions}")
    model_opts = {'text': CustomTextualCLIP(num_classes=nb_emotions,
                                            finetune=args.finetune,
                                            multisentence=True),
                  'vision': CustomVisualCLIP(num_classes=nb_emotions,
                                             finetune=args.finetune)}
    trainer = Trainer(model=model_opts[args.branch], loss_fn=loss,
                      metrics=metrics, monitor_metric=args.monitor,
                      device=args.device, learning_rate=args.lr,
                      lr_decay=args.decay_lr, logger=logger)
    trainer.fit(data_loader=train_loader, val_loader=val_loader,
                max_epochs=args.epochs, patience=args.patience,
                tol_eps=args.eps)
    test_preds, test_loss = trainer.eval(data_loader=test_loader,
                                         use_best=True, verbose=True)
    test_metrics = trainer.assess(data_loader=test_loader,
                                  predictions=test_preds)
    logger(f"Loss: {test_loss}\nMetrics:{test_metrics}")

    if args.save:
        logger("Saving model's state dict in disk.")
        trainer.save(logger.log_dir / "model_state_dict.pt")
    if args.export:
        logger("Exporting model's test predictions to file.")
        numpy.save(logger.log_dir / "test_predictions",
                   test_preds.cpu().numpy())
    logger.close()


if __name__ == "__main__":
    main()
