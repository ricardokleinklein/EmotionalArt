""" Preprocess Artpedia

This script's goal is to provide a set of versions from which neural models
can then be trained on learning the association between images and captions
describing them.

Note: User must explicitly call this script adding the optional flags whose
text style they want to incorporate as captions for the images. An error
will prompt if none is indicated.

Position arguments:
    src                     Dataset's JSON data file
    dst                     Location to save the newly created processed file

Optional arguments:
    --context               Keep contextual captions
    --visual                Keep descriptive captions
    -d, --download          Download images
    -h, --help              Show this help message and exit
"""
import argparse
import pandas
import warnings
import itertools
import requests

from PIL import Image
from unidecode import unidecode
from pathlib import Path
from tqdm import tqdm

from typing import List


ART_DIR = "/mnt/HDD/DATA/Artpedia/images"


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Artpedia JSON data file")
    parser.add_argument("dst", type=str, help="Location to save new file")

    parser.add_argument("--contextual", action="store_true",
                        help="Include contextual captions")
    parser.add_argument("--visual", action="store_true",
                        help="Include visually descriptive captions")

    parser.add_argument("--artdir", type=str, default=ART_DIR,
                        help="Image local directory")
    parser.add_argument("-d", "--download", help="Force image downloading",
                        action="store_true")
    return parser.parse_args()


def download(url: str, dst: Path) -> bool:
    """

    Args:
        url:
        dst:

    Returns:

    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X '
                             '10_11_5) AppleWebKit/537.36 (KHTML, '
                             'like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.ok:
        open(dst, "wb").write(response.content)
    image = Image.open(dst)
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    image_without_exif.save(dst)
    return response.ok


def pick_cc(x: pandas.Series, cols: List[str]) -> str:
    sentences = list(itertools.chain(*[x.__getattr__(col) for col in cols]))
    sentences = [unidecode(s).replace('. ', ' ') for s in sentences]
    return ' '.join(sentences)


def main():
    args = parse_args()
    dataset = pandas.read_json(args.src).transpose()
    dataset['title'] = dataset['title'].apply(lambda s: unidecode(s).lower())

    # Download images if not present, or force downloading again if flag
    artdir = Path(args.artdir)
    if not artdir.exists() or args.download:
        artdir.mkdir(parents=True, exist_ok=True)
        titles = dataset['title'].values
        urls = dataset['img_url'].values
        for i, (title, url) in tqdm(enumerate(zip(titles, urls))):
            status = download(url, artdir / (title + '.jpg'))
            if not status:
                warnings.warn(f"Could not download {title} from {url}",
                              RuntimeWarning)

    # Assess availability and keep only downloaded ones
    dataset['localpath'] = dataset["title"].apply(
        lambda s: artdir / (s + '.jpg'))
    dataset = dataset[dataset['localpath'].map(lambda s: s.exists())]

    # Select captions source
    if not args.contextual and not args.descriptive:
        raise KeyError("Must indicate a source (or both) from: ["
                       "'contextual', 'visual']")
    cc_cols = []
    if args.visual:
        cc_cols.append("visual_sentences")
    if args.contextual:
        cc_cols.append("contextual_sentences")
    dataset['description'] = dataset.apply(
        lambda x: pick_cc(x, cc_cols), axis=1)

    # Sort them alphabetically
    dataset.sort_values(by='year', inplace=True)
    dataset.reset_index(inplace=True)
    dataset.to_csv(args.dst, index=False)


if __name__ == "__main__":
    main()
