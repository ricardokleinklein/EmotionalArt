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
    --artdir        Local directory where images are saved
    -a, --agreement     Filter out paintings whose
    --val           Val size
    --test          Test size
    -s, --seed      Random seed
    -q, --quiet     Hide messages
    -h, --help      Show this help message and exit


"""
import argparse
import numpy
import pandas

from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

pandas.options.mode.chained_assignment = None  # default='warn'


ART_DIR = "/mnt/HDD/DATA/ARTEMIS/artemis_official_data/official_data/wikiart"


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Artemis Data (processed "
                                              "according to the paper)'s CSV")
    parser.add_argument("dst", type=str, help="Location to save new file")
    parser.add_argument("--artdir", type=str, default=ART_DIR,
                        help="Image local directory")
    parser.add_argument("-a", "--agree", type=float, default=0.0,
                        help="Paintings whose principal mode (emotion) has "
                             "less than this probability density will be "
                             "filtered out.")
    parser.add_argument("--val", type=float, default=0.15, help="Val size")
    parser.add_argument("--test", type=float, default=0.25, help="Test size")
    parser.add_argument("-s", "--seed", type=int, default=1234,
                        help="Random seed")
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

    dataset = pandas.read_csv(file)[:100]

    # Check all artworks are available locally
    dataset['localpath'] = dataset.apply(
        lambda x: absolute_local_path(img_root, x.art_style, x.painting),
        axis=1)     # Obtain image paths in the local system
    dataset = dataset[dataset['localpath'].map(lambda s: s.exists())]

    # Append a dot to the end of a sentence if it doesn't have one
    dataset['utterance'] = dataset['utterance_spelled'].apply(
        lambda s: s + '.' if s[-1] != '. ' else s
    )

    # Merging all utterances to a single one for each artwork
    # Also, turn individual emotion labels into probability distributions
    unique_artworks = dataset['painting'].unique()
    merged_utterances = []
    label_bin = LabelBinarizer()
    label_bin.fit(dataset['emotion'].unique())
    emotion_dists = []
    most_emotion = []
    under_agreement_lvl = []
    for artwork in tqdm(unique_artworks, total=len(unique_artworks),
                        disable=args.quiet):
        subset = dataset[dataset['painting'] == artwork]
        if len(subset) < 2: # samples with only 1 utterance
            subset = pandas.concat([subset, subset])

        merged_utterances.append(' '.join(subset['utterance']))
        emotion_count = label_bin.transform(subset['emotion']).sum(axis=0)
        artwork_dist = emotion_count / emotion_count.sum()
        if not numpy.max(artwork_dist) > args.agree:
            under_agreement_lvl.append(subset.index[0])
        most_emotion.append(numpy.argmax(emotion_count))
        emotion_dists.append(artwork_dist)

    dataset.drop_duplicates(subset='painting', inplace=True)
    dataset['utterance'] = merged_utterances
    dataset['emotion'] = emotion_dists
    dataset['emotion_label'] = most_emotion

    # Remove samples whose agreement between annotators is smaller than
    # threshold
    dataset.drop(under_agreement_lvl, inplace=True)

    # Select partitions stratified by majority emotion of the artwork
    dev, test = train_test_split(dataset.index, test_size=args.test,
                                 random_state=args.seed,
                                 stratify=dataset['emotion_label'])
    nb_val = int(args.val * len(dataset))
    train, val = train_test_split(dev, test_size=nb_val,
                                  random_state=args.seed,
                                  stratify=dataset.loc[dev]['emotion_label'])
    dataset['split'].loc[train] = "train"
    dataset['split'].loc[val] = "val"
    dataset['split'].loc[test] = "test"

    # Sort them alphabetically
    dataset.sort_values(by='painting', inplace=True)
    dataset.reset_index(inplace=True)
    dataset.to_csv(savedir, index=False)


if __name__ == "__main__":
    main()
