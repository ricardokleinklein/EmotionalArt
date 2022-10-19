"""
DEPRECATED - Please refer to the HuggingFace's transformer pipeline to call
models.
"""
import numpy

from typing import List, Union
# from sentence_transformers import SentenceTransformer

# Compilation of recommended models at date 15th October 2021.
# The models here collected only include the models trained on the largest
# amount of data from the whole pool.
# Visit https://www.sbert.net/docs/pretrained_models.html for more information.
_pretrained = [
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/all-distilroberta-v1',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/clip-ViT-B-32' # CLIP base model, able to extract embeddings from images
]


class SBert:

    def __init__(self, model: str = _pretrained[0]) -> None:
        """
        Loads a SentenceTransformer model that can be ised to map either
        text or images to embeddings.

        Args:
            model: Specific pretrained model to start from.
        """
        self.name = model
        self.transformer = SentenceTransformer(model)

    def __call__(self, samples: List[Union[str, numpy.ndarray]]) -> \
            numpy.ndarray:
        """
        Compute sentence embeddings.

        Args:
            samples (List[Union[str, numpy.ndarray]]): Inputs

        Returns:
            numpy.ndarray of embeddings
        """
        return self.transformer.encode(samples, show_progress_bar=True)
