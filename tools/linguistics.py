import numpy
import pandas
import string

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union, Set
from nltk import tokenize
from transformers import AutoTokenizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


Series = pandas.Series


def remove_punctuation(text: str, special: str = '.') -> str:
	"""
	TODO
	Args:
		text:
		special:

	Returns:

	"""
	punctuation = string.punctuation.replace(special, '')
	return text.translate(str.maketrans('', '', punctuation))


def tf(documents: Union[str, List[str], Series],
	   top_k: Optional[int] = None) -> Dict[str, str]:
	""" Estimate the frequency of the terms in a set of documents. Words are
	obtained splitting sentences by whitespaces-

	Args:
		documents: Texts to consider.
		top_k: Top terms most represented.

	Returns:
		words and their absolute frequency.
	"""
	if not isinstance(documents, str):
		documents = ' '.join(documents)
	words = documents.split(' ')
	return {x[0]: x[1] for x in Counter(words).most_common(top_k)}


def tf_idf(documents: Dict[int, str], top_k: int) -> \
		Dict[int, List[Union[list, Any]]]:
	"""
	Retrieve the top_k most characteristic words in every document according to
	a TF-IDF measure.

	Args:
		documents (Dict[int, str]):
		top_k (int): Number of relevant words per document.

	Returns:
		Dict[int, List[Union[list, Any]]]
	"""
	vectorizer = TfidfVectorizer(strip_accents='unicode',
								 lowercase=True,
								 stop_words='english')
	docs = [' '.join(val) for key, val in documents.items()]
	vec_docs = vectorizer.fit_transform(docs)
	feature_names = vectorizer.get_feature_names()
	relevant_words = dict()
	for i in range(vec_docs.shape[0]):
		row = vec_docs.getrow(i).toarray()[0]
		k_words = numpy.flip(list(numpy.argsort(row)[-top_k:]))
		relevant_words[i] = [feature_names[j] for j in k_words]
	return relevant_words


class OOVWords:
	"""Compute the ratio of words covered and not covered by the default
	vocabulary a model is pretrained with.

	Attributes:
		tokenizer:
		model_vocabulary:
		text_vocabulary:

	Example:
		>>> oov = OOVWords('openai/clip-vit-base-patch32')
		>>> text = ['this is a sentence', 'another sentence']
		>>> oov(text) # Collects text's vocabulary
	"""

	def __init__(self, pretrained: str) -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

		self.model_vocabulary = set(
			map(self.clean_word, set(self.tokenizer.get_vocab().keys()))
		)
		self.text_vocabulary = set()

	def __call__(self, text: List[str]) -> Set[str]:
		"""Store the set of words that conform a text's vocabulary.

		Args:
			text: List of sentences.

		Returns:
			the set of words that make up a text, its vocabulary.
		"""
		vocab = set()
		for sentence in text:
			tokenized = tokenize.word_tokenize(sentence)
			vocab.update([token for token in tokenized])
		self.text_vocabulary = vocab
		return vocab

	@staticmethod
	def clean_word(word: str) -> str:
		"""Remove special characters such as accents and </w> tags.

		Args:
			word: Word to clean.

		Returns:
			cleaned word
		"""
		return word.replace('</w>', '').replace('Ġ', '')

	def __repr__(self):
		fmt = '       Out-Of-Words Ratio\n'
		fmt += '--------------------------------\n'
		fmt += '   Text Vocabulary Size: {:d}\n'
		fmt += '--------------------------------\n'
		fmt += '   Model Vocabulary Size: {:d}\n'
		fmt += '--------------------------------\n'
		fmt += '     Covered Words %: {:.2f}\n'
		fmt += '     OOV Words %: {:.2f}\n'
		fmt += '--------------------------------\n'
		inter = set.intersection(self.text_vocabulary, self.model_vocabulary)
		diff = set.difference(self.text_vocabulary, self.model_vocabulary)
		return fmt.format(
			len(self.text_vocabulary),
			len(self.model_vocabulary),
			len(inter) / len(self.text_vocabulary),
			len(diff) / len(self.text_vocabulary)
		)


class VaderAnalysis:

	def __init__(self):
		"""
		TODO
		"""
		self.vader = SentimentIntensityAnalyzer()

	def __call__(self, text):
		df = {'sentence': [], 'pos': [], 'neg': [], 'neu': [], 'compound': []}
		for sentence in text:
			sentiment_dict = self.vader.polarity_scores(sentence)
			df['sentence'].append(sentence)
			for key in sentiment_dict:
				df[key].append(sentiment_dict[key])
		df = pandas.DataFrame(df)
		return pandas.DataFrame(df)


class TextBlobAnalysis:

	def __init__(self, subjectivity=True, POS=False):
		"""
		TODO
		"""
		self.textblob = TextBlob

	def __call__(self, text):
		df = {'sentence': [], 'subjectivity': []}
		for sentence in text:
			blob = self.textblob(sentence)
			df['sentence'].append(sentence)
			df['subjectivity'].append(blob.subjectivity)
		return pandas.DataFrame(df)


class BrysbaertConcreteness:

	def __init__(self, path_to_file):
		"""
		TODO
		"""
		self.df = pandas.read_csv(path_to_file, sep='\t')
		self.gram_extractor = TextBlob

	def __call__(self, text):
		df = {'sentence': [], 'concreteness': []}
		for sentence in tqdm(text, total=len(text)):
			sentence_score = []
			df['sentence'].append(sentence)

			bi_grams = self.gram_extractor(sentence).ngrams(n=2)
			for bi_gram in bi_grams:
				idx = self._find_match(bi_gram, bigram=True)
				if idx is not None:
					sentence_score.append(self.df['Conc.M'].loc[idx])
					sentence = sentence.replace(self.df['Word'].loc[idx], '')
			words = sentence.split()
			for word in words:
				idx = self._find_match(word)
				if idx is not None:
					sentence_score.append(self.df['Conc.M'].loc[idx])

			sentence_score = numpy.mean(sentence_score) if sentence_score else None
			df['concreteness'].append(sentence_score)

		return pandas.DataFrame(df)

	def _find_match(self, word, bigram=False):
		"""
		TODO
		"""
		if bigram:
			word = ' '.join(word)
		match = self.df[self.df['Word'] == word]
		if not match.empty:
			return match.index[0]
		return None

