""" Preprocess Artemis

This script is aimed at the structuring and standardizing of the original
Artemis data file.

Because our setup is different from theirs, we do not respect the split of
the data, given our working unit is the individual painting and not the
utterances.

Positional arguments:
    src             File to process
    dst             Location to save the newly created file

Optional arguments:
    --img_root      Local directory where images are saved
    -h, --help      Show this help message and exit


"""
import argparse
import pandas

from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer


ART_DIR = "/mnt/HDD/DATA/ARTEMIS/artemis_official_data/official_data/wikiart"


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original Artemis CSV")
    parser.add_argument("dst", type=str, help="Location to save new file")
    parser.add_argument("--artdir", type=str,
                        help="Image local directory", default=ART_DIR)
    parser.add_argument("-q", "--quiet", help="Hide messages",
                        action="store_true")
    return parser.parse_args()


def absolute_local_path(root: Path, painting_style: str, painting_name: str) \
        -> Path:
    """ Construct the absolute local path to acces the image.

    Args:
        root: Root directory containing the subfolders of images.
        painting_style: Subfolder according to painting style.
        painting_name: Filename.

    Returns:
        Absolute local path of a painting.
    """
    return root / painting_style / (painting_name + '.jpg')


def main() -> None:
    args = parse_args()
    file = Path(args.src)
    savedir = Path(args.dst)
    img_root = Path(args.artdir)
    if not file.exists():
        raise IOError(f"Artemis file not found in {file}")

    dataset = pandas.read_csv(file)
    dataset['utterance'] = dataset['utterance'].apply(
        lambda s: s + '.' if s[-1] != '.' else s
    )   # Append a dot to the end of a sentence if it doesn't have one

    # Check all artworks are available locally
    dataset['localpath'] = dataset.apply(
        lambda x: absolute_local_path(img_root, x.art_style, x.painting),
        axis=1)     # Obtain image paths in the local system
    dataset = dataset[dataset['localpath'].map(lambda s: s.exists())]

    # Merging all utterances to a single one for each artwork
    # Also, turn individual emotion labels into probability distributions
    unique_artworks = dataset['painting'].unique()
    merged_utterances = []
    label_bin = LabelBinarizer()
    label_bin.fit(dataset['emotion'].unique())
    emotion_dists = []
    for artwork in tqdm(unique_artworks, total=len(unique_artworks),
                        disable=args.quiet):
        subset = dataset[dataset['painting'] == artwork]
        merged_utterances.append(''.join(subset['utterance']))
        emotion_count = label_bin.transform(subset['emotion']).sum(axis=0)
        emotion_dists.append(emotion_count / emotion_count.sum())

    dataset.drop_duplicates(subset='painting', inplace=True)
    dataset['utterance'] = merged_utterances
    dataset['emotion'] = emotion_dists

    # Sort them alphabetically
    dataset.sort_values(by='painting', inplace=True)
    dataset.to_csv(savedir, index=False)


if __name__ == "__main__":
    main()
