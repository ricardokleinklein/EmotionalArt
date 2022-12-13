""" Preprocess Artemis

This script is aimed at the structuring and standarding of the original
Artemis data file (once cleaned as indicated in the original paper).

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


ART_DIR = "/mnt/HDD/DATA/ARTEMIS/artemis_official_data/official_data/wikiart"


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original Artemis CSV")
    parser.add_argument("dst", type=str, help="Location to save new file")
    parser.add_argument("--img_root", type=str,
                        help="Image local directory", default=ART_DIR)
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
    img_root = Path(args.img_root)
    if not file.exists():
        raise IOError(f"Artemis file not found in {file}")
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)

    dataset = pandas.read_csv(file)
    dataset['localpath'] = dataset.apply(
        lambda x: absolute_local_path(img_root, x.art_style, x.painting),
        axis=1)
    dataset = dataset[dataset['localpath'].map(lambda s: s.exists())]
    dataset.sort_values('painting', ascending=True, inplace=True,
                        ignore_index=True)
    dataset.to_csv(savedir, index=False)


if __name__ == "__main__":
    main()
