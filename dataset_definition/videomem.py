"""
	Base class of the VideoMem dataset.
"""
import os
import PIL.Image
import numpy
import pandas
from typing import List, Union, TypeVar

from tools.io_pipes import CSVPipe, Mp4Pipe

PandasDataFrame = pandas.DataFrame
Image = PIL.Image.Image


def remove_dashes(astring):
    return astring.replace('-', ' ')


class VideoMem:
    """VideoMem media memorability dataset.

	This dataset is composed of 8000 7-seconds long videos, each of which
	is annotated by a different amount of people according to the
	memorability score achieved in a memory game.

	Attributes:
		root: Root directory where data is stored.
		df: a pandas.DataFrame collecting all the tabular information of
		the dataset.
	"""
    captions_file: str = '{:s}-set_video-captions.txt'
    score_file: str = 'ground-truth_{:s}-set.csv'
    video_dir: str = 'Videos'

    def __init__(self, root: str) -> None:
        self.root = root
        substr = f'{self.__class__.__name__} from {root}'
        if os.path.isfile(root):
            print('Loading dataset ' + substr)
            self.df = CSVPipe().read(root)
        else:
            print('Building dataset ' + substr)
            self.df = self.build_corpus_dataframe()

        self.df = self.df.dropna()

    def _collect_dataset(self, partition: str) -> PandasDataFrame:
        """
		Extend all tabular data from a partition to form a single dataframe.

		Args:
			partition: Set of data to work on.

		Returns:
			dataframe with all data fields.
		"""
        text_df = self.get_source_text(partition)
        score_df = self.get_source_scores(partition, size=len(text_df))
        return pandas.concat([text_df, score_df], axis=1)

    def get_source_text(self, partition: str) -> PandasDataFrame:
        """
		Compile descriptions from source file.

		Args:
			partition: Set of data to work on.

		Returns:
			video name and sentences (1 per video).
		"""
        df = CSVPipe().read(os.path.join(
            self.root, partition + '-set', self.captions_file.format(
                partition)), header=None, sep='\t',
            names=['video', 'description'])
        df['description'] = df['description'].apply(remove_dashes)
        return df

    def get_source_scores(self, partition: str, size: int = None) -> \
            PandasDataFrame:
        """

		Args:
			partition:
			size:

		Returns:

		"""
        cols = ['video', 'short-term_memorability',
                'nb_short-term_annotations', 'long-term_memorability',
                'nb_long-term_annotations']
        if partition != 'test':
            score_df = pandas.read_csv(os.path.join(
                self.root, partition + '-set',
                self.score_file.format(partition)
            ), usecols=cols[1:])
        else:
            score_df = pandas.DataFrame(None, index=numpy.arange(size),
                                        columns=cols[1:])
        return score_df

    def build_corpus_dataframe(self) -> PandasDataFrame:
        """
		Extend all tabular data from the corpus into a single dataframe.

		Returns:
			Body of the corpus in tabular format.
		"""
        dev_df = self._collect_dataset('dev')
        test_df = self._collect_dataset('test')
        df = pandas.concat([dev_df, test_df], axis=0, ignore_index=True)
        split = numpy.repeat(['dev', 'test'], [len(dev_df), len(test_df)])
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
        if index:
            condition = self.df['video'] == 'video' + str(index) + '.webm'
            return self.df[condition]['description']
        if partition:
            return self.df[self.df['partition'] == partition]['description']
        return self.df['description']

    def get_scores(self, timeterm: str = 'short') -> PandasDataFrame:
        """
		Return the set of scores.

		Args:
			timeterm: Short or Long-term scoring.

		Returns:
			Score per sample
		"""
        return self.df[f'{timeterm}-term_memorability'].dropna()

    def get_sample_frames(self, index: int, fps: int = None, **kwargs) -> \
            List[Image]:
        """
		Extract the frames of a video clip.

		Args:
			index: Video name to retrieve frames from.
			fps: Frames Per Second.

		Returns:
			Images from a given video clip.
		"""
        item = self.df[self.df['video'] == f'video{index}.webm']
        if item.empty:
            item = self.df[self.df['video'] == index]  # Complete video name
        video_path = os.path.join(
            self.root, item['partition'].values[0] + '-set', self.video_dir,
            item['video'].values[0])
        keep_temp = kwargs.get('keep_temp', False)
        frames = Mp4Pipe().read(video_path, fps=fps,
                                keep_temp=keep_temp, **kwargs)
        return frames

    def get_c3d(self, partition: str) -> PandasDataFrame:
        """Retrieve C3D features previously computed.

        Args:
            partition: Whether to extract exclusively from a partition.

        Returns:
            C3D vector per sample

        NOTE: Provisional version of this method.
        """
        feats_dir = os.path.join(self.root, partition + '-set', 'features',
                                 'C3D')
        subset = self.df[self.df['partition'] == partition]
        feats = []
        for clip in subset['video']:
            c3d_feats = numpy.genfromtxt(os.path.join(feats_dir,
                                                      clip.replace('webm',
                                                                   'txt')),
                                         delimiter=' ')
            feats.append(c3d_feats)
        return numpy.array(feats)

    def export(self, to_file: str, **kwargs) -> None:
        """Save the dataset to an external file.

		Args:
			to_file: File to export to.

		"""
        CSVPipe().write(to_file, self.df, **kwargs)
