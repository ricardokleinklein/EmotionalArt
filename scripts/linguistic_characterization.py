"""
Linguistic Characterization

Characterize the texts describing artworks in a specific dataset.

A) Grammar category proportion / sentence
B) Idiosyncracy: Sentiment compound, Subjectivity, Concreteness

Positional arguments:
    src                 Dataset source csv file
    col                 Text column name in data
"""
import argparse
import nltk
import pandas
import string

from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize


def parse_dataset() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Dataset source csv file")
    parser.add_argument("col", type=str, help="Text column name in data")
    return parser.parse_args()


def table_parse(texts: pandas.Series) -> pandas.DataFrame:
    """ Estimate the proportion per sentence of most relevant grammar
    categories.

    Args:
        texts: Texts to analyze.

    Returns:
        Proportion per category
    """
    classes = {'words': [], 'nouns': [], 'pronouns': [],
               'adjectives': [], 'adverbs': [], 'adpositions': [], 'verbs': []}
    for sens in tqdm(texts):
        sens = sent_tokenize(sens)
        for s in sens:
            s = s.translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(s)
            tagged = nltk.pos_tag(words, tagset='universal')
            nouns = [x for x in tagged if 'NOUN' in x[1]]
            pronouns = [x for x in tagged if 'PRON' in x[1]]
            adjectives = [x for x in tagged if 'ADJ' in x[1]]
            adverbs = [x for x in tagged if 'ADV' in x[1]]
            adpos = [x for x in tagged if 'ADP' in x[1] or '']
            verbs = [x for x in tagged if 'VERB' in x[1]]
            classes['words'].append(len(words))
            classes['nouns'].append(len(nouns))
            classes['pronouns'].append(len(pronouns))
            classes['adjectives'].append(len(adjectives))
            classes['adverbs'].append(len(adverbs))
            classes['adpositions'].append(len(adpos))
            classes['verbs'].append(len(verbs))
    return pandas.DataFrame(classes)


def main():
    args = parse_dataset()
    dataset = pandas.read_csv(args.src)
    stats = table_parse(dataset[args.col])
    print(f"For {args.src},")
    print(stats.mean(axis=0))


if __name__ == "__main__":
    main()
