""" Preprocess SemArt

This script is aimed at the structuring and standardizing of the original
Semart data files, and oriented towards a CLIP-like training scheme.

Positional arguments:
    src             Directory with train/val/test splits
    dst             Location to save the newly created file

Optional arguments:
    --img_root      Local directory where images are saved
    -h, --help      Show this help message and exit


"""
import argparse
import pandas

from tqdm import tqdm
from pathlib import Path


ART_DIR = "/mnt/HDD/DATA/SEMART/Images"


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original raw data directory")
    parser.add_argument("dst", type=str, help="Location to save new file")
    parser.add_argument("--clip", action="store_true",
                        help="Preprocess to make it simpler for a "
                             "CLIP-schemed training.")
    parser.add_argument("--img_root", type=str,
                        help="Image local directory", default=ART_DIR)
    parser.add_argument("-q", "--quiet", help="Hide messages",
                        action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rootdir = Path(args.src)
    savedir = Path(args.dst)
    img_root = Path(args.img_root)

    raw_files = list(rootdir.glob("*.csv"))
    if not rootdir.exists() or len(raw_files) == 0:
        raise IOError(f"SemArt files not found in {rootdir}")
    dataset = pandas.concat(
        [pandas.read_csv(
            f, sep='\t', encoding="ISO-8859-1") for f in raw_files],
        ignore_index=True)
    dataset['localpath'] = dataset["IMAGE_FILE"].apply(
        lambda s: img_root / s)
    dataset = dataset[dataset['localpath'].map(lambda s: s.exists())]
    dataset.sort_values(by="TITLE", inplace=True)

    if args.clip:
        dataset_ = {k: [] for k in list(dataset)}
        nb = len(dataset)
        for i in tqdm(range(nb), total=nb, disable=args.quiet):
            painting = dataset.iloc[i]
            sentences = painting["DESCRIPTION"].split('.')
            for sentence in sentences:
                for key in list(dataset_):
                    if key == "DESCRIPTION":
                        dataset_[key].append(sentence)
                    else:
                        dataset_[key].append(painting[key])
        dataset = pandas.DataFrame(dataset_)

    dataset.to_csv(savedir, index=False)


if __name__ == "__main__":
    main()
