"""

"""
import argparse
import torch
import pandas
import torch.nn as nn

from data_preprocess.datasets import CLIPDataset
from neural_models.transformers import CLIP
from metrics.multilabel_classification import BatchWiseClsMetrics
from workflow.trainer import Trainer
from loggers.baselog import Logger


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original dataset CSV")
    parser.add_argument("--val", action="store_true",
                        help="If set, leave out data for validation")
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


class ContrastiveLoss(nn.Module):

    def __init__(self, batch_size: int, device: torch.device) -> None:
        """

        Args:
            batch_size:
        """
        super(ContrastiveLoss, self).__init__()
        self.length = batch_size
        self.device = device
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        gt = torch.arange(self.length).to(self.device)
        left = self.loss_img(input[0], gt)
        right = self.loss_txt(input[1], gt)
        return (left + right) / 2


def main():
    # Read command line arguments
    args = parse_args()
    dataset = pandas.read_csv(args.src)[:15]
    loss = ContrastiveLoss(args.batch, args.device)
    metrics = BatchWiseClsMetrics()
    logger = Logger(log_dir=args.log_dir)
    logger(f"Experiment configuration:\n"
           f" {[f'{k}={v!r}' for k, v in args.__dict__.items()]}")

    if args.val:
        train_data = dataset[dataset['split'] == "train"]
        test_data = dataset[dataset['split'] == "val"]
    else:
        train_data = dataset[dataset['split'].isin(["train", "val"])]
        test_data = dataset[dataset['split'] == "test"]
    logger(f"\n\tNb train samples: {len(train_data)}\n\tNb eval samples: "
           f"{len(test_data)}")

    train = CLIPDataset(train_data, text_col="utterance",
                        image_col="localpath")
    train_loader = train.load("train", batch_size=args.batch)
    test = CLIPDataset(test_data, text_col="utterance", image_col="localpath")
    test_loader = test.load("test", batch_size=args.batch)
    model = CLIP()

    trainer = Trainer(model=model, loss_fn=loss, metrics=metrics,
                      monitor_metric=args.monitor, multimodal=True,
                      device=args.device, learning_rate=args.lr, logger=logger)
    record = trainer.fit(data_loader=train_loader,
                         max_epochs=args.epochs, patience=args.patience,
                         tol_eps=args.eps)
    logger.close()


if __name__ == "__main__":
    main()
