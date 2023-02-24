"""
Wrappers around text tokenizers.

Useful in order to manually select tokens to add to a dictionary,
or to modify the default embeddings or rules within.

# TODO: Add methods to retrain tokenizers from scratch.
"""
from transformers import BertTokenizer, GPT2Tokenizer, CLIPTokenizer
from typing import Dict, List, Optional


class BPETokenizer:
    """ General class for Byte-Pair Encoding tokenizers, including
    WordPiece.
    """

    TOKENIZERS = {
        'gpt2': (GPT2Tokenizer, 'gpt2'),
        'clip': (CLIPTokenizer, 'openai/clip-vit-base-patch32'),
        'wordpiece': (BertTokenizer, 'bert-base-uncased')
    }

    def __init__(self, name_or_path: Optional[str],
                 seq_len: Optional[int] = None) -> None:
        self.name = name_or_path
        self.tokenizer = self.TOKENIZERS[name_or_path][0].from_pretrained(
            self.TOKENIZERS[name_or_path][1]
        )
        self.tokenizer.add_prefix_space = True
        self.tokenizer.use_cache = False
        if seq_len is not None:
            self.tokenizer.model_max_length = seq_len
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self) -> int:
        return len(self.tokenizer)

    def __call__(self, sentence, **kwargs) -> Dict:
        return self.tokenizer(sentence, **kwargs)

    def __add__(self, new_tokens: List) -> None:
        self.tokenizer.add_tokens(new_tokens)
