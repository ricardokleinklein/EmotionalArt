""" Aesthetics

In this script we finetune the classifier from a pretrained visual encoder
in order to learn the mapping between the visual features of artworks and
the average artwork rating in terms of liking as seen by a pool of annotators.

Positional arguments:
    src                     Dataset file.

Optional arguments:
    --pretrained              Path to pretrained model state dict
    --val                   Whether to train according to val error
    -e, --epochs            Epochs to train for
    -p, --patience          Epochs to wait before early-stopping
    -b, --batch             Batch size
    --lr                    Initial learning rate
    --eps                   Error tolerance for early-stopping
    --log-dir               Experiment dirname to label logs under
    -s, --save              Whether to save trained models afterwards
    --seed                  Random seed value
    --device               Device to use. Default: cuda
"""
import argparse
import pandas
import pathlib

from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch.nn import MSELoss
from data_preprocess.datasets import ImageDataset
from neural_models.transformers import *
from metrics.regression_metrics import RegressionMetrics
from workflow.trainer import Trainer
from loggers.baselog import Logger


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str,
                        help="Processed (Aesthetics) Wikiart CSV")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights")
    parser.add_argument("--val", action="store_true",
                        help="If set, evaluate over val data. test otherwise")
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


def retrieve_pretrained_vision(path_to_pretrained: pathlib.Path,
                               num_classes: int = 1,
                               device: str = "cuda") -> Module:
    """ Perform our custom loading of a branch from a pretrained CLIP state
    dict.

    Args:
        path_to_pretrained: Path to trained model's state dict
        num_classes: Renewed final layer output size
        device: Which device to state_dict to

    Returns:
        a branch of a CLIP model whose weigths are retrieved from a
        checkpoint but for the final layer.
    """
    # read pretrained state dict
    pretrained_state_dict = torch.load(path_to_pretrained,
                                       map_location=torch.device(device))
    full_clip = CLIP()
    full_clip.load_state_dict(pretrained_state_dict, strict=True)
    model = full_clip.__getattr__("vision")

    # replace last layer (classifier)
    model.classifier = nn.Linear(
        in_features=getattr(model, "base_vision_proj").out_features,
        out_features=num_classes
    )

    # freeze all layers but final classifier
    frozen_layers = [l for l in [getattr(model, "base_vision_proj"), getattr(
        model, "base_vision_clip")]]
    for layer in frozen_layers:
        for param in layer.parameters():
            param.requires_grad = False
    model.output_embed = False
    return model


def main():
    args = parse_args()
    wikiart = pandas.read_csv(args.src)
    metrics = RegressionMetrics()
    print(list(wikiart))

    logger = Logger(log_dir=args.log_dir)
    logger(f"Experiment configuration:\n"
           f" {[f'{k}={v!r}' for k, v in args.__dict__.items()]}")

    if args.val:
        train_data = wikiart[wikiart['overlap'] == True]
        train_data, test_data = train_test_split(train_data, test_size=0.2)
    else:
        train_data = wikiart[wikiart['overlap'] == True]
        test_data = wikiart[wikiart['overlap'] != True]

    logger(f"\n\tNb train samples: {len(train_data)}\n\tNb eval samples: "
           f"{len(test_data)}")
    train = ImageDataset(train_data['localpath'].values,
                        scores=train_data['Ave. art rating'].values)
    train_loader = train.load(phase="train", batch_size=args.batch)
    test = ImageDataset(test_data['localpath'].values,
                       scores=test_data['Ave. art rating'].values)
    test_loader = test.load(phase="test", batch_size=args.batch)
    val_loader = test_loader if args.val else None

    model = retrieve_pretrained_vision(args.pretrained, device=args.device)
    loss = MSELoss()
    trainer = Trainer(model=model, loss_fn=loss,
                      metrics=metrics, monitor_metric="loss",
                      device=args.device, learning_rate=args.lr, logger=logger)

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
        trainer.save(logger.log_dir / "model_state_dict.pt", use_best=True)
    logger.close()


if __name__ == "__main__":
    main()
