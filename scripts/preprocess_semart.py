""" Preprocess SemArt

This script is aimed at the structuring and standardizing of the original
Semart data files, and oriented towards a CLIP-like training scheme.

Positional arguments:
    src             Directory with train/val/test splits
    dst             Location to save the newly created file

Optional arguments:
    --add-field     Fields to add as sentences
    --img_root      Local directory where images are saved
    -h, --help      Show this help message and exit

"""
import re
import argparse
import pandas

from unidecode import unidecode
from pathlib import Path


ART_DIR = "/mnt/HDD/DATA/SEMART/Images"


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original raw data directory")
    parser.add_argument("dst", type=str, help="Location to save new file")
    parser.add_argument("--add-field", nargs='+', default=None,
                        help="Extra metada to add as part of the description")
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

    raw_files = list(rootdir.glob("semart*.csv"))
    if not rootdir.exists() or len(raw_files) == 0:
        raise IOError(f"SemArt files not found in {rootdir}")
    dataset = pandas.concat(
        [pandas.read_csv(
            f, sep='\t', encoding="ISO-8859-1") for f in raw_files],
        ignore_index=True)
    dataset.dropna(subset=["DESCRIPTION"], inplace=True)
    dataset['localpath'] = dataset["IMAGE_FILE"].apply(
        lambda s: img_root / s)
    dataset = dataset[dataset['localpath'].map(lambda s: s.exists())]
    dataset.sort_values(by="TITLE", inplace=True)

    # Correct and uniform unicode characters
    dataset["DESCRIPTION"] = dataset["DESCRIPTION"].apply(
        lambda s: unidecode(re.sub(r'(?<=[.,])(?=[^\s])', r' ', s)) + '. ')
    dataset["AUTHOR"] = dataset["AUTHOR"].apply(lambda s: unidecode(s))
    dataset["TITLE"] = dataset["TITLE"].apply(lambda s: unidecode(s))

    if args.add_field is not None:
        extra_sentence = dataset[args.add_field].apply(
            lambda s: ', '.join(s) + ".", axis=1
        )
        dataset["DESCRIPTION"] = dataset["DESCRIPTION"] + extra_sentence
    dataset["DESCRIPTION"] = dataset["DESCRIPTION"].apply(
        lambda s: re.sub('"','',s).lower())
    dataset.to_csv(savedir, index=False)


if __name__ == "__main__":
    main()
