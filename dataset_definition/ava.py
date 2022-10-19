"""
AVA: Aesthetic Visual Analysis Dataset - Murray et al (2012)

Definiton of a class able to clean its data, sort them and split in training
and testing partitions. When used as a stand-alone script, this
script creates an instance of the dataset and saves it in an external file
according to the configuration submitted as arguments.

Input:
    delta: AVA parameter that removes samples with 5 +- delta aesthetic score.
"""
import os
import numpy
import pandas
import requests
import sys
import time

from bs4 import BeautifulSoup
from os.path import join
from PIL import Image
from typing import Union, List
from tqdm import tqdm

from tools.io_pipes import CSVPipe


def starts(astring: str, letter: str) -> bool:
    """Checks whether a string begins with a given letter."""
    return astring.startswith(letter)


def avg_score(votes: numpy.ndarray) -> numpy.ndarray:
    """ Weighted-average of votes for scores 1 to 10.

    Args:
        votes (numpy.ndarray): Votes per score.

    Returns:
        ndarray weighted-average
    """
    return numpy.average(numpy.linspace(1, 10, 10), weights=votes)


def weighted_var(votes: numpy.ndarray) -> numpy.ndarray:
    """Weighted variance given a series of votes for 1-10 scores.

    Args:
        votes (numpy.ndarray): Votes per score.
    Returns:
        numpy.ndarray
    """
    average = avg_score(votes)
    variance = numpy.average((numpy.linspace(1, 10, 10) - average)**2,
                             weights=votes)
    return numpy.sqrt(variance)


class DPChallengeScrapper:
    base_url = 'https://www.dpchallenge.com/'
    chall_url = 'challenge_results.php?CHALLENGE_ID='
    wait_time = 60

    def __init__(self):
        """Web scrapping targeted at capturing textual descriptions from
        challenges in dpchallenge.com"""
        self.count = 0  # Number of challenges visited

    def __call__(self, challenge_id: int):
        """
        Retrieve the textual description of a challenge.

        Args:
            challenge_id: Challenge index.

        Returns:
            str
        """
        if not isinstance(challenge_id, str):
            challenge_id = str(challenge_id)
        page = requests.get(join(self.base_url, self.chall_url + challenge_id))
        soup = BeautifulSoup(page.content, "html.parser")
        # Get exclusively the description space of the html file
        desc = " ".join(soup.body.find_all('div')[3].text.split()[1:])
        self._pause()
        self.count += 1
        return desc

    @staticmethod
    def _pause():
        time.sleep(60)


class Ava:
    """AVA: Aesthetic Visual Analysis Dataset - Murray et al (2012).

        Dataset composed by 25000 images extracted from DPChallenge.com,
        a social media platform in which users vote on their favourite
        photographs weekly for a set of open challenges. Amateur and
        professional submissions are accepted alike.

        From the original paper, given that most of the samples have an
        average score, a margin `delta` is incorporated to the selection of
        data points so those samples whose score is above 5 + delta are
        regarded as 'beautiful' whereas those below 5 - delta are considered
        'ugly'.

        Attributes:
            root: Dataset root directory.
    """
    ava_file = 'AVA.txt'  # Main file of the dataset information
    tags_file = 'tags.txt'  # Semantics tag # and meaning
    challenges_file = 'challenges-desc.txt'  # Challenge # and title

    def __init__(self, root: str) -> None:
        self.root = root
        substr = f'{self.__class__.__name__} from {root}'
        if os.path.isfile(root):
            print('Loading dataset ' + substr)
            self.df = CSVPipe().read(root)
        else:
            print(f'Building dataset ' + substr)
            root_file = os.path.join(root, self.ava_file)
            self.df = CSVPipe().read(root_file, header=None, sep=' ')
            # Keep only those samples with readable images
            failed_samples = self.validate_images()
            self.remove_samples(failed_samples)
            self.add_additional_info_files(descriptions=True)

    def challenge_info(self, descriptions: bool = False, fill_missing=True,
                       inplace=False) -> Union[pandas.DataFrame, pandas.Series]:
        """ Incorporate the information related to the specific challenge
        each photograph was submitted to.

        Args:
            descriptions (bool): Whether to include challenge description.
            fill_missing (bool): Whether to complete missing descriptions in
                challenges with more than one edition.
            inplace (bool): Whether to modify the inner df.

        Returns:
            pandas.DataFrame or pandas.Series with photo-wise challenge info.
        """
        challenge_file = pandas.read_csv(join(self.root,
                                              self.challenges_file),
                                         header=None, sep=' ')
        challenge_file.set_index(0, inplace=True)

        if fill_missing and descriptions:
            challenge_file = self.fill_edition_missing(challenge_file)

        title = self.df[14].map(challenge_file[1])
        if not descriptions:
            if inplace:
                self.df = pandas.concat([self.df, title], axis=1)
                self.df.columns = [*self.df.columns[:-1], 'chall-title']
            return title
        desc = self.df[14].map(challenge_file[2])
        challenge_info = pandas.concat([title, desc], axis=1,
                                       ignore_index=True)
        if inplace:
            self.df = pandas.concat([self.df, challenge_info], axis=1)
            self.df.columns = [*self.df.columns[:-2], 'chall-title',
                               'chall-desc']
        return challenge_info

    def semantic_info(self, inplace=False) -> pandas.DataFrame:
        """
        Incorporate the information related to the two semantic tags every
        sample in the dataset is labelled with.

        Args:
            inplace (bool): Whether to modify the inner df.

        Returns:
            pandas.DataFrame with the name of the semantic labels.
        """
        with open(join(self.root, self.tags_file), 'r') as f:
            file = f.read().split('\n')
        file = [tuple(x.split(maxsplit=1)) for x in file[:-1]]
        tags = pandas.DataFrame.from_records(file, columns=['tag', 'name'])
        tags['tag'] = pandas.to_numeric(tags['tag'])
        tag_1 = self.df[12].map(tags.set_index('tag')['name'])
        tag_2 = self.df[13].map(tags.set_index('tag')['name'])
        tags_info = pandas.concat([tag_1, tag_2], axis=1, ignore_index=True)
        if inplace:
            self.df = pandas.concat([self.df, tags_info], axis=1)
            self.df.columns = [*self.df.columns[:-2], 'tag_1', 'tag_2']
        return tags_info

    def weighted_scores(self, inplace=False) -> pandas.Series:
        """Compute the weighted-average score for every sample from the
        distribution of votes.

        Args:
            inplace (bool): Whether to modify the inner df.

        Returns:
            pandas.Series of scores
        """
        vote_cols = list(range(2, 12))
        weighted = self.df[vote_cols].apply(avg_score, axis=1)
        if inplace:
            self.df = pandas.concat([self.df, weighted], axis=1)
            self.df.columns = [*self.df.columns[:-1], 'score']
        return weighted

    def weighted_variance(self, inplace=False) -> pandas.Series:
        """Compute the weighted variance for every sample from the
        distribution of votes.

        Args:
            inplace (bool): Whether to modify the inner df.

        Returns:
            pandas.Series of variance values.
        """
        vote_cols = list(range(2, 12))
        weighted = self.df[vote_cols].apply(weighted_var, axis=1)
        if inplace:
            self.df = pandas.concat([self.df, weighted], axis=1)
            self.df.columns = [*self.df.columns[:-1], 'variance']
        return weighted

    def add_additional_info_files(self, descriptions: bool = True) -> None:
        """Extend inplace the original ava summary file with the data coming
        from challenge, semantics and scoring information.

        Args:
            descriptions (bool): Whether to include challenge descriptions.
        """
        self.weighted_scores(inplace=True)
        self.weighted_variance(inplace=True)
        self.semantic_info(inplace=True)
        self.challenge_info(descriptions, fill_missing=True, inplace=True)

    def scrap_challenge_descriptions(self) -> pandas.DataFrame:
        """Scrap challenge's websites to read their description.

        Returns:
            pandas.DataFrame: new challenge info

        NOTE: I made this function quickly and left an error handling
        procedure to manage potential errors, but didn t care about the
        actual errors.
        """
        original = pandas.read_csv(join(self.root, self.challenges_file),
                                   sep=' ', header=None)
        scrapper = DPChallengeScrapper()
        descriptions = []
        ids = original[0]
        for i, item in enumerate(tqdm(ids)):
            try:
                print(scrapper(item))
                descriptions.append(scrapper(item))
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(f"Error happened: {e}")
                descriptions.append('ERROR-SCRAPPER')
        original['description'] = descriptions
        return original

    def validate_images(self) -> List[int]:
        """
        Check out whether images in the dataset are readable.
        Notice this is an expensive operation.

        Returns:
            List[int]: Indices of the dataset not readable.
        """
        idx2rm = []
        length = len(self.df)
        for i in tqdm(range(length), total=length):
            try:
                self.load_image(i)
            except FileNotFoundError:
                idx2rm.append(i)
        return idx2rm

    def load_image(self, index: int = None, ident: int = None) -> Image:
        """
        Load an image either by its index within the main df or by picture's
        name.

        Args:
            index (int): Index within self.df.
            ident (int): Identifier of the photograph, name of the picture.

        Returns:
            Pillow.Image
        """
        path = join(self.root, 'images')
        if index is not None:
            img_path = join(path, str(self.df.iloc[index][1]) + '.jpg')
            return Image.open(img_path)
        img_name = self.df[self.df[1] == ident][1].values[0]
        img_path = join(path, str(img_name) + '.jpg')
        return Image.open(img_path)

    def remove_samples(self, indices: List[int]) -> None:
        """
            Remove samples by index in the main dataframe.

        Args:
            indices (List[int]): Indices of the samples to remove.
        """
        self.df.drop(index=indices, inplace=True)

    def filter_by_margin(self, delta: int) -> None:
        """
        Filter out inplace samples in between 5-delta < score < 5 + delta.

        Args:
            delta (int): Scoring margin.
        """
        gtdelta = self.df[self.df['score'] > (5 + delta)]
        stdelta = self.df[self.df['score'] < (5 - delta)]
        self.df = pandas.concat([stdelta, gtdelta], axis=0, ignore_index=True)

    def get_scores(self) -> pandas.Series:
        """
        Retrieves the distributions of scores in the dataset.

        Returns:
            pandas.Series: Unnormalized scoring distribution.
        """
        if 'score' in list(self.df):
            return self.df['score']
        return self.weighted_scores()

    def _get_challenge_description(self, index: int = None) -> str:
        """
        Read the description of the challenge a photo was submitted to.
        Args:
            index (int): Index within self.df.

        Returns:
            str
        """
        if 'chall-desc' in list(self.df):
            return self.df.iloc[index]['chall-desc']
        return self.challenge_info(True).iloc[index][1]

    def get_description(self, index: int = None) -> str:
        """Return descriptions in the dataset, unless a particular sample is
        specified.

        Args:
            index: Optional, whether to select a particular sample index.

        Returns:
            Challenge descriptions
        """
        if not index:
            return self.df['chall-desc']
        self._get_challenge_description(index)

    def fill_edition_missing(self, challenges_df) -> pandas.DataFrame:
        """Duplicate challenge descriptions for those repeated editions
        without their own description.

        Args:
            challenges_df (pandas.DataFrame): Titles and descriptions.

        Returns:
            pandas.DataFrame
        """
        null_desc = challenges_df[challenges_df[2].isnull()]
        null_indices = null_desc.index.values
        complete = challenges_df.dropna(subset=[2])
        new_df = {1: [], 2: []}
        for i in range(len(null_desc)):
            title_i = null_desc[1].iloc[i].split('_')
            last = title_i[-1]
            if not (len(title_i) == 1) and (starts(last, 'I') or starts(
                    last, 'V')):
                title_i = title_i[:-1]

            src = '_'.join(title_i)
            src_desc = challenges_df[challenges_df[1] == src][2].values
            # Special case: Closest to Textures_VI -> Textures_II
            if not src_desc:
                src_desc = challenges_df[challenges_df[1] == 'Textures_II']
                src_desc = src_desc[2].values
            src_desc = src_desc[0]
            # Challenges with explicit titles
            if not isinstance(src_desc, str):
                src_desc = src

            new_df[1].append(null_desc[1].iloc[i])
            new_df[2].append(src_desc)
        new_df = pandas.DataFrame(new_df, index=null_indices)
        return pandas.concat([complete, new_df], axis=0)

    def export(self, to_file: str, **kwargs) -> None:
        """Save the dataset to an external file.

        Args:
            to_file: File to export to.

        """
        CSVPipe().write(to_file, self.df, **kwargs)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self):
        return self.df.__repr__()


if __name__ == "__main__":
    delta = 0
    if len(sys.argv) > 1:
        delta = int(sys.argv[1])
    rootdir = '/media/ricardokleinlein/HDD/DATA/AVA/'
    corpus = Ava(rootdir)
    corpus.export('./ava.csv')

