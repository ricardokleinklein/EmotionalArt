""" Preprocess WikiArt-Emotions

Positional arguments:
    src                     Directory to read data info files from
    dst                     Location to save processed data

Optional arguments:
    --artdir                Directory to save downloaded images
    -q, --quiet             Mute messages
    -h, --help              Display additional information
"""
import argparse
import pandas
import requests

from pathlib import Path
from tqdm import tqdm


WIKIART_INFO = "WikiArt-info.tsv"
WIKIART_ANNOTATIONS = "WikiArt-Emotions-All.tsv"
ART_DIR = "/mnt/HDD/DATA/WikiArt-Emotions/paintings"


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="WikiArt-Emotions TSV")
    parser.add_argument("dst", type=str,
                        help="Location to save processed data")

    parser.add_argument("--artdir", type=str, default=ART_DIR,
                        help="Image local directory")
    parser.add_argument("--artemis", type=str, default=None,
                        help="Artemis data file. If set, annotate overlap")
    parser.add_argument("-q", "--quiet", help="Hide messages",
                        action="store_true")
    return parser.parse_args()


def absolute_local_path(root: Path, artist: str, title: str, year: str) \
        -> Path:
    """ Construct the absolute local path to access the image.

    Args:
        root: Root directory containing the subfolders of images.
        artist: Artist's name.
        title: Artwork's title.
        year: Artwork's year of creation

    Returns:
        Absolute local path of a painting.
    """
    filename = '-'.join([s.replace(' ', '_') for s in [artist, title, year]])
    filename = filename.replace('/', '_')
    return root / (filename.lower() + ".jpg")


def download(url: str, dst: Path) -> bool:
    """

    Args:
        url:
        dst:

    Returns:

    """
    response = requests.get(url)
    if response.ok:
        try:
            open(dst, "wb").write(response.content)
        except requests.exceptions.ConnectionError or TimeoutError:
            print(RuntimeWarning(f"Could not download {url}"))
    return response.ok


def decompose_url(url: str) -> str:
    """ Remove domain of the url and make it agree with artemis format.

    Args:
        url: Painting url.

    Returns:
        painting's name as shown in Artemis Dataset.
    """
    st = Path(url)
    return str(st.parent.name) + '_' + str(st.name)


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    download_dir = Path(args.artdir)
    download_dir.mkdir(parents=True, exist_ok=True)
    wikiart = pandas.read_csv(src / WIKIART_INFO, sep='\t')
    annotations = pandas.read_csv(src / WIKIART_ANNOTATIONS, sep='\t')
    annotations = annotations[['ID', 'Ave. art rating']]

    dataset = wikiart.merge(annotations, on='ID')
    drop_cols = [s for s in list(dataset) if ':' in s]
    dataset.drop(columns=drop_cols, inplace=True)

    # Check all artworks are available locally
    dataset['localpath'] = dataset.apply(
        lambda x: absolute_local_path(
            download_dir, x.Artist, x.Title, x.Year), axis=1)
    dataset['in_local'] = dataset['localpath'].map(lambda s: s.exists())

    for i, painting in tqdm(dataset.iterrows(),
                            total=len(dataset), disable=args.quiet):
        if not painting['in_local']:
            success = download(painting['Image URL'], painting['localpath'])

    dataset = dataset[dataset['localpath'].map(lambda s: s.exists())]

    if args.artemis:
        artemis = pandas.read_csv(args.artemis)
        dataset['painting'] = dataset['Painting Info URL'].apply(decompose_url)
        overlap = set(artemis['painting']).intersection(dataset['painting'])
        dataset['overlap'] = dataset['painting'].apply(
            lambda x: any([k in x for k in overlap])
        )
    dataset.to_csv(args.dst, index=False)


if __name__ == "__main__":
    main()
