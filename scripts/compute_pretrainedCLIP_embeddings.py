"""Compute_pretrainedCLIP_embeddings

This script can be used to easily compute embeddings to either textual and
image samples from pretrained models as hosted in HuggingFace.
"""

import argparse
import numpy
import pandas
import pathlib
import shutil

from tqdm import tqdm
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import CLIPProcessor, CLIPVisionModel

from typing import Dict, Optional, Union

FROM_HUB = "openai/clip-vit-base-patch32"
TOKENIZER = CLIPTokenizer.from_pretrained(FROM_HUB, do_lower_case=True)
PROCESSOR = CLIPProcessor.from_pretrained(FROM_HUB)
BASEMODEL_TXT = CLIPTextModel.from_pretrained(FROM_HUB)
BASEMODEL_VIS = CLIPVisionModel.from_pretrained(FROM_HUB)


def through_clip(text: Optional[str] = None, image: Optional[str] = None,
                 only_vectors: bool = True) -> Union[numpy.ndarray, Dict]:
    """Compute embeddings from a pretrained CLIP model, either for a
    sentence or an image, but one at a time.

    Args:
        text: Sentence.
        image: Image path.
        only_vectors: Whether to return the complete output or only the
        sample's vector embedding.

    Returns:
        a 512-D vector embedding
    """
    if not text:
        inputs = PROCESSOR(images=Image.open(image), return_tensors="pt")
        output = BASEMODEL_VIS(**inputs)
    if not image:
        inputs = TOKENIZER(text, return_tensors="pt", truncation=True,
                           max_length=77)
        output = BASEMODEL_TXT(**inputs)
    if only_vectors:
        return output.pooler_output.squeeze().detach().cpu().numpy()
    return output


def iterate_dataset(modality: str, data: pandas.Series,
                    delete_after: bool = True, tmp: pathlib.Path = "tmp"
                    ) -> None:
    """ Compute embeddings from pretrained models.

    Args:
        modality: Modality to compute.
        data: Source of data to compute embeddings for.
        delete_after: Remove tmp directory with computed embeddings.
        tmp: Name of the directory within which embeddings are saved.
    """
    ds_length = len(data)
    tmp = pathlib.Path(tmp)
    if not tmp.exists():
        pathlib.Path.mkdir(tmp, parents=True, exist_ok=True)
        for i in tqdm(range(ds_length), total=ds_length):
            sample = data.iloc[i]
            name = data.index[i]
            x = through_clip(**{modality: sample})
            numpy.save(tmp / name, x)
    if delete_after:
        shutil.rmtree(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=str,
                        help="Pandas DataFrame containing datasets' "
                             "information")
    parser.add_argument("save-dir", type=str, help="Directory within "
                                                   "which computed "
                                                   "samples will be stored")
    parser.add_argument("modality", type=str, help="Modality to compute",
                        choices=["text", "image"])
    parser.add_argument("col", type=str, help="Column name from dataset")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    file = pathlib.Path(args.file)
    savedir = pathlib.Path(args.save_dir)
    column = args.col

    dataset = pandas.read_csv(file) if file.exists() else IOError
    iterate_dataset(modality=args.modality, data=dataset[column],
                    delete_after=False, tmp=savedir)


if __name__ == "__main__":
    main()
