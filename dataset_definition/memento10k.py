"""
Memento10K - Newman et al. (2020)

Definition of a class able to clean this dataset, sort it and split in the
different partitions it contains.

When used as a stand-alone script, creates an instance of the dataset and
saves it in an external storing file ready to be read by other applications.
"""

import os
import PIL
import numpy
import pandas
from typing import List, Optional

from os.path import join
from tools.io_pipes import CSVPipe, Mp4Pipe


DataFrame = pandas.DataFrame
Series = pandas.Series
Image = PIL.Image.Image


class Memento10K:
	"""Memento10K - Newman et al. (2020).

	Dataset made of 10000 short video clips of about 3-seconds long. The
	original paper describing it explains the annotation procedure, being it
	basically identical to other memory games. The key difference is that
	authors found that memorability follows a log-linear decay with time,
	so they measure exclusively short-term memorability and computed an
	estimation of this decay for every video.

	Scores are normalized in every case. This score is a time interpolation
	from the probability of a video clip to be remembered after a while from
	the first view by a pool of annotators. It is not a raw score because
	the fact that different viewers watched every video in a different
	sequence position requires experimenters to unify a video's score to a
	comparable magnitude.

	Attributes:
		root: Dataset root directory.
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

	def build_corpus_dataframe(self) -> DataFrame:
		"""Agglutinate information distributed among files into a single
		dataframe object."""
		url_dfs = self.get_source_urls()
		description_dfs = self.get_source_descriptions()
		score_dfs = self.get_source_scores()

		part_only = lambda idx: [x[idx] for x in [url_dfs, description_dfs,
												 score_dfs]]
		train_df = pandas.concat(part_only(0), axis=1)
		dev_df = pandas.concat(part_only(1), axis=1)

		# test as a special case
		nb_fields = len(list(score_dfs[0])) - 2	# How many score info fields?
		nb_items = len(url_dfs[2])	# How many test samples?
		score_dfs[-1] = pandas.DataFrame(None, index=numpy.arange(nb_items),
			columns=list(score_dfs[0][-nb_fields:]))
		score_dfs[-1]['video_id'] = url_dfs[-1]['video_id']
		score_dfs[-1]['video_url'] = url_dfs[-1]['video_url']
		test_df = pandas.concat(part_only(2), axis=1)

		df = pandas.concat([train_df, dev_df, test_df], axis=0,
						   ignore_index=True)
		return df.T.drop_duplicates().T

	def get_description(self, index: int = None, partition: str = None) -> \
			DataFrame:
		"""Return descriptions unless a specific one is sought.

		Args:
			index: Optional, specific video_id.
			partition: Whether to extract only a partition.

		Returns:
			Descriptions of video(s).
		"""
		cols = [col for col in self.df if 'description' in col]
		descriptions = self.df[cols].agg(' '.join, axis=1)
		if index:
			return descriptions[self.df['video_id'] == index]
		if partition:
			return descriptions[self.df['partition'] == partition]
		return descriptions

	def get_scores(self, raw: bool = False, partition: str = None) -> \
			DataFrame:
		"""Retrieves the distribution of scores in the dataset.

		Args:
			raw: Whether to retrieve raw or normalized scores.
			partition: Whether to extract only a partition.

		Returns:
			Scores per sample.
		"""
		col = 'scores_{:s}_short_term'
		col = col.format('raw') if raw else col.format('normalized')
		if partition:
			return self.df[self.df['partition'] == partition][col]
		return self.df[col].dropna()

	def get_sample_frames(self, index: int, fps: int = None, **kwargs) -> \
			List[Image]:
		"""Extract the frames of a video clip.

		Args:
			index: Video_id to retrieve frames from.
			fps: Frames Per Second.

		Returns:
			Image frames from a video clip.
		"""
		item = self.df[self.df['video_id'] == index]
		video_path = join(self.root, item['partition'].values[0] + '_set',
						  'Videos', '{:05d}.mp4').format(index)
		keep_temp = kwargs.get('keep_temp', False)
		frames = Mp4Pipe().read(video_path, fps=fps,
												keep_temp=keep_temp, **kwargs)
		return frames

	def get_c3d(self, partition: str) -> DataFrame:
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
											  '{:05d}.mp4.csv'.format(
												  clip)), delimiter=',')
			feats.append(c3d_feats)
		return numpy.array(feats)

	def has_wavfile(self, wav_dir: str, inplace: bool = False) -> Optional[
		DataFrame]:
		""" Filter out samples with no audio wav file.

		Args:
			wav_dir: Location of audio files in disk.
			inplace: Whether to modify this object or a copy.

		Returns:
			Dataset updated, removing no-audio samples and adding a column
			to audios' absolute paths (``wav_path``).
		"""
		name_ = self.df['video_url'].apply(lambda url: url.split('/')[-2:])
		filepaths = name_.apply(lambda s: os.path.join(
			wav_dir, s[0].replace('+', '-') + '_' + s[1] + '.wav'))
		actual_files = filepaths.apply(os.path.isfile)
		tmp = self.df[actual_files]
		tmp['wav_path'] = filepaths[actual_files]
		if inplace:
			self.df = tmp
			return None
		return tmp

	def _over_partitions(func):
		"""Decorator to iterate through the different data partitions."""
		partitions = ['train', 'dev', 'test']

		def wrapper(self, *args, **kwargs):
			output = []
			for p in partitions:
				output.append(func(self, p, *args, **kwargs))
			return output

		return wrapper

	@_over_partitions
	def get_source_urls(self, partition: str) -> List[DataFrame]:
		"""Retrieve URLs, video_id and partitions from files.

		Args:
			partition: Which partition (train/dev/test) to read from.

		Returns:
			A pandas.DataFrame for each partition.
		"""
		df = CSVPipe().read(os.path.join(
			self.root, partition + '_set', self.urls_file.format(partition)))
		df['partition'] = [partition] * len(df)
		return df

	@_over_partitions
	def get_source_descriptions(self, partition: str) -> List[DataFrame]:
		"""Retrieve video_id and all the descriptions attached to videos.

		Args:
			partition: Which partition (train/dev/test) to read from.

		Returns:
			A dataframe for each partition.
		"""
		if hasattr(self, 'df'):
			cols = [col for col in self.df if 'desc' in col] + ['video_id']

		df = CSVPipe().read(os.path.join(
			self.root, partition + '_set', self.captions_file.format(partition)
		))
		return df

	@_over_partitions
	def get_source_scores(self, partition: str) -> Optional[DataFrame]:
		"""Retrieve video_id and scoring info for all videos.

		Args:
			partition: Which partition (train/dev/test) to read from.

		Returns:
			A dataframe for each partition.

		"""
		if partition != 'test':
			df = CSVPipe().read(os.path.join(
				self.root,
				partition + '_set',
				self.score_file.format(partition)
			))
			return df
		return None

	def export(self, to_file: str, **kwargs) -> None:
		"""Save the dataset to an external file.

		Args:
			to_file: File to export to.
		"""
		CSVPipe().write(to_file, self.df, **kwargs)


if __name__ == "__main__":
	rootdir = '/media/ricardokleinlein/HDD/DATA/MediaEval_2021/'
	rootdir += 'Official_release/Memento10k/'
	corpus = Memento10K(rootdir)