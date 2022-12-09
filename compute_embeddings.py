import numpy
import pandas
import pathlib
import shutil

from tqdm import tqdm
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import CLIPProcessor, CLIPVisionModel

from typing import Dict, List, Optional, Union

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",
                                          do_lower_case=True)
model_base = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#model_base = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")


def compute_clip(text: Optional[str] = None,
                 image: Optional[str] = None) -> numpy.ndarray:
    """ Compute pretrained CLIP embeddings for either a sentence or an image.

    TODO: Implement image loading and preprocessing.

    Args:
        text: Sentence to encode
        image: Image.

    Returns:
        a 512-dimensional embedding.
    """

    if text is not None:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=77)
    if image is not None:
        inputs = processor(images=Image.open(image), return_tensors="pt")

    output = model_base(**inputs).pooler_output.squeeze()
    return output.detach().cpu().numpy()


def compute_embeddings(modal: str,
                       data: Union[pandas.Series, List],
                       col: str,
                       delete_after: bool = True,
                       tmp: str = "tmp") -> numpy.ndarray:
    """ Compute embeddings from pretrained models.

    Args:
        modal: Modality to compute.
        data: Source of data to compute embeddings for.
        delete_after: Remove tmp directory with computed embeddings.
        tmp: Name of the directory within which embeddings are saved.
    """
    length = len(data)
    tmp = pathlib.Path(tmp)
    if not tmp.exists():
        pathlib.Path.mkdir(tmp, parents=True, exist_ok=True)
        for i in tqdm(range(length), total=length):
            sample = data[col].iloc[i]
            name = str(data.index[i])
            x = compute_clip(**{modal: sample})
            numpy.save(tmp / name, x)
    embeddings = [numpy.load(tmp / (str(idx) + '.npy')) \
                  for idx in data.index]
    if delete_after:
        shutil.rmtree(tmp)
    return embeddings

artemis_path = pathlib.Path("../artemis/CLEAN_ARTEMIS/artemis_preprocessed.csv")
artemis = pandas.read_csv(artemis_path) if artemis_path.exists() else IOError

folder = "/mnt/HDD/PROJECTS/EmotionalArt/CLIP-pretrained-embeddings/artemis" \
         "-text/embeddings-clean-artemis"
text_x = compute_embeddings(modal='text',
                       data=artemis,
                       col='utterance_spelled',
                       delete_after=False,
                       tmp=folder)
print('Completed!')
