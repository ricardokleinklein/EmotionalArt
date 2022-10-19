"""Access to the official data releases to the MediaEval 2021.

In this occasion the Predicting Media Memorability Challenge, hosted within
the MediaEval workshop.


    Typical usage example:

    trecvid = Trecvid('./TrecVid DATA/')
    captions = trecvid.get_descriptions()
    # Do operations over captions
    X = do_something_over_captions(captions)

"""
import os
import copy
import numpy
import pandas
from typing import List, Union, Tuple, TypeVar, Iterable

from os.path import join

from tools.io import CSVPipe, Mp4Pipe
from custom_datasets import torch_datasets


PandasDataFrame = TypeVar(Union['pandas.core.frame.Series',
                                'pandas.core.frame.DataFrame'])
Image = TypeVar('PIL.Image')

class Trecvid:
    """Subset of the TRECVid dataset (https://trecvid.nist.gov/)

    Subset of short videos annotated with short & long-term memorability
    scores. In both cases, scores have a raw version and a normalized one.
    The raw one corresponds to the proportion of annotators that
    successfully remembered having watched a clip, whereas the normalized
    score is computed as a linear rectification over the raw data, assuming
    the hypothesis posed in Newman et al. (2020).

    Attributes:
        root: a string denoting the root directory of the data, or the file
            to load from.
        df: a pandas.DataFrame collecting all the tabular information of
            the dataset.
    """
    annotation_file = '{:s}_{:s}_term_annotations.csv'
    captions_file = '{:s}_text_descriptions.csv'
    score_file = '{:s}_scores.csv'
    urls_file = '{:s}_video_urls.csv'

    def __init__(self, root: str) -> None:
        self.root = root
        substr = f'{self.__class__.__name__} from {root}'
        if os.path.isfile(root):
            print('Loading dataset ' + substr)
            self.df = CSVPipe().read(root)
        else:
            print('Building dataset ' + substr)
            self.df = self.build_corpus_dataframe()

    def _collect_dataset(self, partition: str = 'train') -> PandasDataFrame:
        """
        Extend all tabular data from a partition into a single DataFrame.

        Args:
            partition: Whether to collect test or train data.
        """
        base_path = join(self.root, partition + '_set')
        df = pandas.read_csv(join(base_path, self.urls_file.format(
            partition)), index_col='video_id')
        desc = self.get_description(partition=partition)
        cols = ['video_id', 'annotations_short_term',
                'annotations_long_term', 'scores_raw_short_term',
                'scores_raw_long_term', 'scores_normalized_short_term',
                'decay_alpha']
        if partition != 'test':
            score = pandas.read_csv(join(base_path, self.score_file.format(
                partition)), usecols=cols, index_col='video_id')

        # There are not any scoring for test, so we fake some null values
        elif partition == 'test':
            score = pandas.DataFrame(None, index=numpy.arange(len(df)),
                                     columns=cols[1:]).set_index(df.index)
        else:
            raise KeyError
        return pandas.concat([df, desc, score], axis=1).reset_index(level=0)

    def build_corpus_dataframe(self) -> PandasDataFrame:
        """
        Extend all tabular data from the corpus into a single DataFrame.

        Returns:
            Body of the corpus in table format.
        """
        train_df = self._collect_dataset('train')
        dev_df = self._collect_dataset('dev')
        test_df = self._collect_dataset('test')
        df = pandas.concat([train_df, dev_df, test_df], axis=0,
                           ignore_index=True)
        split = numpy.repeat(['train', 'dev', 'test'], [len(train_df),
                                                        len(dev_df),
                                                        len(test_df)])
        df['partition'] = split
        return df

    def get_description(self, index: int = None, partition: str = None) -> \
            PandasDataFrame:
        """
        Return the descriptions of a set of videos with their ID.

        Args:
            index: Optional, specific video_id.
            partition: string denoting whether to clip a specific subset.

        Returns:
            Description of video(s).
        """
        if not hasattr(self, 'df'):
            base_path = join(self.root, partition + '_set')
            desc = pandas.read_csv(join(base_path, self.captions_file.format(
                partition)), usecols=['video_id', 'description'])
            # Each video has 2+ caption
            desc = desc.groupby(['video_id'])['description'].apply(
                lambda x: '. '.join(x.astype(str)).lower())
            return desc

        if index:
            return self.df[self.df['video_id'] == index]['description']
        if partition:
            return self.df[self.df['partition'] == partition]['description']

        return self.df['description']

    def get_scores(self, timeterm: str = 'short', partition: str = None,
                   raw: bool = False) -> pandas.DataFrame:
        """
        Return the set of different scores for a subset of the data.

        Args:
            timeterm: Short- or long-term scores.
            raw: Whether to retrieve raw or normalized scores.
            partition: Whether to extract exclusively from a partition.

        Returns:
            Scores per sample
        """
        col = 'scores_{:s}'
        col = col.format('raw') if raw else col.format('normalized')
        col += '_{:s}_term'.format(timeterm)
        if partition:
            return self.df[self.df['partition'] == partition][col]
        return self.df[col].dropna()

    def get_sample_frames(self, index: int, fps: int = None, **kwargs) -> \
            Iterable[Iterable[float]]:
        """Extract the frames of a video clip.

        Args:
            index: Video_id to retrieve frames from.
            fps: Frames Per Second.

        Returns:
            Images from a given video clip.
        """
        item = self.df[self.df['video_id'] == index]
        video_path = join(self.root, item['partition'].values[0] + '_set',
                          'Videos', '{:05d}.mp4').format(index)
        keep_temp = kwargs.get('keep_temp', False)
        frames = Mp4Pipe().read(
            video_path, fps=fps, keep_temp=keep_temp, **kwargs)
        return frames

    def get_c3d(self, partition: str) -> PandasDataFrame:
        """Retrieve C3D features previously computed.

        Args:
            partition: Whether to extract exclusively from a partition.

        Returns:
            C3D vector per sample

        NOTE: Provisional version of this method.
        """
        feats_dir = join(self.root, partition + '_set', 'Features', 'C3D')
        subset = self.df[self.df['partition'] == partition]
        feats = []
        for clip in subset['video_id']:
            c3d_feats = numpy.genfromtxt(join(feats_dir,
                                              '{:05d}.mp4.csv'.format(clip)),
                                         delimiter=',')
            feats.append(c3d_feats)
        return numpy.array(feats)

    def export(self, to_file: str, **kwargs) -> None:
        """Save the dataset to an external file.

        Args:
            to_file: File to export to.

        """
        CSVPipe('csv').write(to_file, self.df, **kwargs)


class TextShortTermTrecvid(torch_datasets.TextDataset):
    def __init__(self, root: str, partition: str = None) -> None:
        self.df = Trecvid(root).df
        super().__init__(
            data=self.df,
            identifier='video_id',
            text='description',
            label='scores_normalized_short_term'
        )

    def __call__(self, partition: str = None) -> None:
        """Select a partition of the dataset."""
        self.df = self.df[self.df['partition'] == partition]
        # TODO: Replace with DummyDataset's idea.
        return self


class TextLongTermTrecvid(torch_datasets.TextDataset):
    def __init__(self, root: str) -> None:
        self.df = Trecvid(root).df
        super().__init__(
            data=self.df,
            text='description',
            label='scores_raw_long_term'
        )


class DebugDataset(torch_datasets.TextDataset):
    def __init__(self, root) -> None:
        self.df = pandas.DataFrame({
            'sample_id': list(range(100)),
            'description': ['a_' + str(i) for i in range(100)],
            'label': numpy.linspace(0.01, 1, 100),
            'partition': ['train'] * 50 + ['dev'] * 50
        })
        super().__init__(
            data=self.df,
            identifier='sample_id',
            text='description',
            label='label'
        )

    def __call__(self, partition: str = None) -> torch_datasets:
        """Select a partition of the dataset."""
        copied_dataset = copy.deepcopy(self)
        copied_dataset.df = self.df[self.df['partition'] == partition]
        return copied_dataset


class DebugMTDataset(torch_datasets.TextDataset):
    def __init__(self, root: str = None) -> None:
        self.df = pandas.DataFrame({
            'sample_id': list(range(100)),
            'description': ['a_' + str(i) for i in range(100)],
            'label': numpy.linspace(0.01, 1, 100),
            'label_2': numpy.linspace(0.01, 1, 100),
            'partition': ['train'] * 50 + ['dev'] * 50
        })
        super().__init__(
            data=self.df,
            identifier='sample_id',
            text='description',
            label=['label', 'label_2']
        )

    def __call__(self, partition: str = None) -> torch_datasets:
        """Select a partition of the dataset."""
        copied_dataset = copy.deepcopy(self)
        copied_dataset.df = self.df[self.df['partition'] == partition]
        return copied_dataset


if __name__ == "__main__":
    rootdir = '/media/ricardokleinlein/HDD/DATA/MediaEval_2021/'
    rootdir += 'Official_release/TRECVid/'
