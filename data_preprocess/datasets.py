from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from PIL import Image
from typing import Tuple, Dict, List, Union

import transformers
import torch
import numpy
import pandas
import glob

# Type definition
Array = numpy.ndarray
DataFrame = pandas.DataFrame
Loader = torch.utils.data.DataLoader
Series = pandas.Series
Tensor = torch.Tensor
Tokenizer = transformers.PreTrainedTokenizer


class SentencesDataset(Dataset):

    default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __init__(self,
                 texts: Union[Array, List, Series],
                 scores: Union[Array, List, Series],
                 tokenizer: Tokenizer = None):
        """Default dataset used for regression over sentences.

        Args:
            texts: Array of texts of the corpus.
            scores: Array of ground-truth labels to compare with.
            tokenizer: Method to split words (BPE-scheme-like).
        """
        self.texts = texts
        self.targets = scores
        self.tokenizer = self.default_tokenizer if tokenizer is None else tokenizer

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx) -> Tuple[Dict, float]:
        """Includes tokenization of the sample."""
        sentence = self.texts[idx]
        target = torch.tensor(self.targets[idx]).float()
        tokenized = self.tokenizer(sentence, return_tensors="pt",
                                   padding="max_length")
        tokenized = {k: val.squeeze() for k, val in tokenized.items()}
        return tokenized, target

    def load(self, phase: str = 'train', batch_size: int = 32, num_workers:
        int = 4) -> Loader:
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
                              shuffle=True, num_workers=num_workers)
        return DataLoader(dataset=self, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)


class VideoByFrameDataset(Dataset):

    valid_formats = {'jpg', 'jpeg', 'png', 'bmp'}
    default_preprocessor = transformers.CLIPFeatureExtractor()

    def __init__(self, videoframes_path, scores, preprocessor = None):
        self.paths = videoframes_path
        self.targets = scores
        self.preprocessor = self.default_preprocessor if \
            preprocessor is None else preprocessor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        video_path = self.paths[idx]
        frame_paths = sorted([f for f in glob.glob(video_path + '*') if \
                              f.split('.')[-1] in self.valid_formats])
        target = torch.tensor(self.targets[idx]).float()
        frames = [Image.open(frame) for frame in frame_paths]
        inputs = self.preprocessor(images=frames, return_tensors='pt')
        return inputs, target

    def load(self, phase: str = 'train', batch_size: int = 1,
             num_workers: int = 0):
        if phase == "train":
            return DataLoader(dataset=self, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)
