""" Retrieve Predictions

Load data and a model to compute predictions from (helpful in those cases
for which we couldn't compute those predictions online)

Positional arguments:
    src                     Dataset file.

Optional arguments:
    --val                   Whether to train according to val error
    -f, --finetune          If set, train only last layer
    -b, --batch             Batch size
    --loss                  Loss function to use ["kldiv", "emd"]
    --log-dir               Experiment dirname to label logs under
    --seed                  Random seed value
    --device               Device to use. Default: cuda

"""
import argparse
import numpy
import pandas
import torch
from pathlib import Path
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
                            'processor': None}}


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
    parser.add_argument("run", type=str, help="Pretrained model dir")
    parser.add_argument("-f", "--finetune", action="store_true",
                        help="If True, train only classification layer(s)")
    parser.add_argument("-b", "--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--loss", type=str, default='kldiv',
                        choices=['kldiv', 'emd', 'entropy'],
                        help="Loss function")
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
    loss = LOSS[args.loss]

    metrics = DistributionDistanceMetrics()

    test_data = artemis[artemis['split'] == "test"]

    data_reader = BRANCH_CONFIG[args.branch]['reader']
    feat_col = BRANCH_CONFIG[args.branch]['feat_col']
    processor = BRANCH_CONFIG[args.branch]['processor']

    test = data_reader(test_data[feat_col].values,
                       scores=tonumpy(test_data['emotion']),
                       processor=processor)
    test_loader = test.load(phase="test", batch_size=args.batch)

    nb_emotions = len(artemis['emotion_label'].unique())
    model_opts = {'text': CustomTextualCLIP(num_classes=nb_emotions,
                                            finetune=args.finetune,
                                            multisentence=True),
                  'vision': CustomVisualCLIP(num_classes=nb_emotions,
                                             finetune=args.finetune)}
    model = model_opts[args.branch]
    pretrained_state_dict = torch.load(Path(args.run) / "model_state_dict.pt",
                                   map_location=torch.device(args.device))
    model.load_state_dict(pretrained_state_dict, strict=True)
    trainer = Trainer(model=model, loss_fn=loss, monitor_metric="loss",
                      device=args.device)
    test_preds, test_loss = trainer.eval(data_loader=test_loader,
                                         use_best=True, verbose=True)
    numpy.save(Path(args.run) / "test_predictions2", test_preds.cpu().numpy())


if __name__ == "__main__":
    main()
