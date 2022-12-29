from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from PIL import Image
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BatchEncoding
from transformers import CLIPFeatureExtractor, CLIPProcessor
from typing import Tuple, Dict, List, Union, Any, Optional

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

    def __getitem__(self, idx) -> Tuple[BatchEncoding, Any]:
        """Includes tokenization of the sample."""
        sentences = sent_tokenize(self.texts[idx])
        target = torch.tensor(self.targets[idx]).float()
        tokenized = self.tokenizer(sentences, return_tensors="pt",
                                   padding="max_length", truncation=True)
        return tokenized, target

    def load(self, phase: str = 'train', batch_size: int = 32,
             num_workers: int = 0) -> torch.utils.data.DataLoader:
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

    def load(self, phase: str = 'train', batch_size: int = 32,
             num_workers: int = 0) -> torch.utils.data.DataLoader:
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


class MultimodalDataset(Dataset):
    default = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def __init__(self, texts: Union[numpy.ndarray, pandas.Series],
                 image_paths: Union[numpy.ndarray, pandas.Series],
                 scores: Union[numpy.ndarray, List],
                 processor: Optional[transformers.ProcessorMixin] = None
                 ) -> None:
        """

        TODO: This is phase 2 of experimentation: merging of modalities.
        Finish CLIP pretraining first.

        Args:
            texts:
            image_paths:
            scores:
            processor:
        """
        self.targets = scores
        self.processor = self.default if not processor else processor

        if isinstance(texts, pandas.Series):
            texts = texts.values
        if isinstance(image_paths, pandas.Series):
            image_paths = image_paths.values
        self.texts = texts
        self.img_paths = image_paths

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx) -> Tuple[Dict, torch.Tensor]:
        sentences = sent_tokenize(self.texts[idx])
        image = Image.open(self.img_paths[idx])
        target = torch.tensor(self.targets[idx]).float()
        inputs = self.processor(text=sentences, images=image,
                                return_tensors="pt", padding="max_length",
                                truncation=True)
        return inputs, target

    def load(self, phase: str = 'train', batch_size: int = 32,
             num_workers: int = 0) -> torch.utils.data.DataLoader:
        """Retrieve a DataLoader to ease the pipeline.

        Args:
            phase: Whether it's train or test.
            batch_size: Samples per batch.
            num_workers: Cores to use.

        Returns:
            an iterable torch DataLoader.
        """
        pass


# https://github.com/pytorch/pytorch/blob/e5742494f6080c8e6f43c37689fc18a7c4b39dfd/torch/utils/data/dataloader.py#L145
class BalancedBatchSampler(BatchSampler):

    def __init__(self, labels: torch.Tensor, n_classes: int, n_samples: int):
        """BatchSampler - from a MNIST-like dataset, samples n_classes and
        within these classes samples n_samples. Gives back batches of size
        n_classes * n_samples.
    """
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: numpy.where(self.labels.numpy() == label)[0] for label in
            self.labels_set}
        for l in self.labels_set:
            numpy.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = numpy.random.choice(self.labels_set, self.n_classes,
                                          replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                    self.used_label_indices_count[class_]:
                    self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[
                    class_] + self.n_samples > len(
                    self.label_to_indices[class_]):
                    numpy.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class CLIPDataset(Dataset):
    default = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __init__(self, data: pandas.DataFrame, text_col: str, image_col: str,
                 processor: transformers.PreTrainedTokenizer = None) -> None:
        """

        """
        self.captions = []
        self.image_paths = []
        self.processor = self.default if not processor else processor

        for _, (img_path, sentences) in data[[image_col, text_col]].iterrows():
            sents = sent_tokenize(sentences)
            for s in sents:
                self.image_paths.append(img_path)
                self.captions.append(s)

        self.image_paths_set = list(data[image_col])
        self.path2label = {path: self.image_paths_set.index(path) for path in
                           self.image_paths_set}

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Tuple[
        Tuple[Dict[str, Any], dict], Tensor]:
        img_path = self.image_paths[idx]
        # img_path = "/Users/ricardokleinlein/Desktop/Thesis/EmotionalArt/DATA" \
        #            "/artemis/wikiart/Baroque/adriaen-brouwer_in-the-tavern-1.jpg"
        image = Image.open(img_path).convert("RGB")
        caption = self.captions[idx]
        label = torch.tensor(self.path2label[img_path])
        # label = torch.tensor(0)
        inputs = self.processor(text=caption, images=image,
                                return_tensors="pt", padding="max_length",
                                truncation=True)
        txt_feats = {k: inputs[k].squeeze() for k in ['input_ids',
                                                      'attention_mask']}
        vis_feats = {'pixel_values': inputs['pixel_values'].squeeze()}
        return (vis_feats, txt_feats), label

    def load(self, phase: str = 'train', batch_size: int = 32,
             num_workers: int = 0) -> torch.utils.data.DataLoader:
        """Retrieve a DataLoader to ease the pipeline.

        Args:
            phase: Whether it's train or test.
            batch_size: Samples per batch.
            num_workers: Cores to use.

        Returns:
            an iterable torch DataLoader.
        """
        labels = torch.tensor(list(range(len(self.image_paths))))
        sampler = BalancedBatchSampler(labels, batch_size, 1)
        return DataLoader(dataset=self,  num_workers=num_workers,
                          batch_sampler=sampler)
