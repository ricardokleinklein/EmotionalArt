from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from transformers import CLIPFeatureExtractor
from typing import Tuple, Dict, List, Union, Optional

import transformers
import torch
import numpy
import pandas


def multisentence_collate(data):
    target = torch.stack([x[1] for x in data])
    sentence_batch = [x[0] for x in data]
    return sentence_batch, target


class SentencesDataset(Dataset):

    default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __init__(self,
                 texts: Union[numpy.ndarray, List, pandas.Series],
                 scores: Union[numpy.ndarray, List, pandas.Series],
                 processor: transformers.PreTrainedTokenizer = None):
        """Default dataset used for regression over sentences.

        Args:
            texts: Array of texts of the corpus.
            scores: Array of ground-truth labels to compare with.
            processor: Method to split words (BPE-scheme-like).
        """
        self.texts = texts
        self.targets = scores
        self.tokenizer = self.default_tokenizer if processor is None else \
            processor

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx) -> Tuple[Dict, float]:
        """Includes tokenization of the sample."""
        sentences = sent_tokenize(self.texts[idx])
        target = torch.tensor(self.targets[idx]).float()
        tokenized = self.tokenizer(sentences, return_tensors="pt",
                                   padding="max_length", truncation=True)
        #tokenized = {k: val for k, val in tokenized.items()}
        return tokenized, target

    def load(self, phase: str = 'train', batch_size: int = 32, num_workers:
        int = 0) -> torch.utils.data.DataLoader:
        """Retrieve a DataLoader to ease the pipeline.

        Args:
            phase: Whether it's train or test.
            batch_size: Samples per batch.
            num_workers: Cores to use.

        Returns:
            an iterable torch DataLoader.
        """
        if phase == "train":
            return DataLoader(dataset=self, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=multisentence_collate)
        return DataLoader(dataset=self, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=multisentence_collate)

class ImageDataset(Dataset):

    default = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def __init__(self,
                 image_paths: Union[numpy.ndarray, List],
                 scores: Union[numpy.ndarray, List],
                 processor: transformers.FeatureExtractionMixin = None
                 ) -> None:
        """

        Args:
            image_paths:
            scores:
            processor:
        """
        self.paths = image_paths
        self.targets = scores
        self.processor = self.default if processor is None else processor

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx) -> Tuple[Dict, float]:
        """

        Args:
            idx:

        Returns:

        """
        image = Image.open(self.paths[idx])
        target = torch.tensor(self.targets[idx]).float()
        pixels = self.processor(image, return_tensors='pt')
        pixels = {k: val.squeeze() for k, val in pixels.items()}
        return pixels, target

    def load(self, phase: str = 'train', batch_size: int = 32, num_workers:
        int = 0) -> torch.utils.data.DataLoader:
        """Retrieve a DataLoader to ease the pipeline.

        Args:
            phase: Whether it's train or test.
            batch_size: Samples per batch.
            num_workers: Cores to use.

        Returns:
            an iterable torch DataLoader.
        """
        shuffle = True if phase == "train" else False
        return DataLoader(dataset=self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
