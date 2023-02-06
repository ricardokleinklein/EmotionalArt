""" Precompute embeddings

This script helps to speed up the process of experimentation. Once a CLIP
model has been finetuned over a specific dataset, we can precompute
embeddings from it over a new set of data, so training of a classifier is
carried out exclusively over the last layer, far more efficient.


Positional arguments:
    src                     Dataset file

Optional arguments:
    pretrained              Path to pretrained model's state dict

"""
import argparse
import numpy
import pandas
import pathlib

import torch.cuda
from tqdm import tqdm
from torch.nn import Module
from data_preprocess.datasets import MultimodalDataset
from neural_models.transformers import *


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original dataset CSV")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights")
    return parser.parse_args()


def from_pretrained(path_to_pretrained: pathlib.Path) -> Module:
    """ Perform our custom loading of a branch from a pretrained CLIP state
    dict.

    Args:
        path_to_pretrained: Path to trained model's state dict

    Returns:
        a CLIP model whose weigths are retrieved from a checkpoint.
    """
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    full_clip = CLIP()
    full_clip.output_embed = True
    full_clip.textual.multiple = True
    if path_to_pretrained is None:
        return full_clip
    pretrained_state_dict = torch.load(path_to_pretrained,
                                       map_location=torch.device(device))
    full_clip.load_state_dict(pretrained_state_dict, strict=True)
    return full_clip


def main():
    args = parse_args()
    dataset = pandas.read_csv(args.src)
    if 'index' in list(dataset):
        dataset.set_index('index', inplace=True)

    model = from_pretrained(args.pretrained)

    data_reader = MultimodalDataset(texts=dataset["utterance"].values,
                                    image_paths=dataset['localpath'].values,
                                    scores=[0.0] * len(dataset),
                                    processor=None)
    if args.pretrained is None:
        args.pretrained = "./fake-parent-dir"
    vector_dir = pathlib.Path(args.pretrained).parent / "precomputed"
    vector_dir.mkdir(exist_ok=True, parents=False)
    for i, x in enumerate(tqdm(data_reader)):
        name = str(dataset.index[i])
        with torch.no_grad():
            inputs = ({'pixel_values': x[0]['pixel_values']},
                    [{'input_ids': x[0]['input_ids'],
                    'attention_mask': x[0]['attention_mask']}])
            v = model(inputs)
            numpy.save(file=vector_dir / name, arr=v.detach().cpu().numpy())


if __name__ == "__main__":
    main()
