"""
Transformer models used by the interfase of HuggingFace.
"""

import torch
import torch.nn as nn

from transformers import BertModel, CLIPModel, GPT2Model
from transformers import PreTrainedTokenizer
from typing import Dict


Tensor = torch.Tensor
Tokenizer = PreTrainedTokenizer


class CustomTextualCLIP(nn.Module):

    def __init__(self, num_classes: int, finetune: bool = False,
                 multisentence: bool = False):
        """
        Visual half of a CLIP model ready to fine-tune on classification
        and/or regression tasks.

        Args:
            num_classes: Labels to predict.
            finetune: If True, adjust only the last layer, otherwise train
            the whole model.

        """
        super(CustomTextualCLIP, self).__init__()
        clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.multiple = multisentence
        self.base_text_clip = clip.text_model
        self.base_text_proj = clip.text_projection
        if finetune:
            for layer in [self.base_text_clip, self.base_text_proj]:
                for param in layer.parameters():
                    param.requires_grad = False
        self.classifier = nn.Linear(self.base_text_proj.out_features,
                                    num_classes)

    def forward(self, x: Dict) -> Tensor:
        if self.multiple:
            z = self._forward_multiple(x)
        else:
            z = self._simple(x)
        z = self.classifier(z)
        return nn.functional.log_softmax(z, dim=1)

    def _forward_multiple(self, x: Dict) -> Tensor:
        """Forward pass splitting in different sentences within each sample."""
        z = [torch.mean(self._simple(x_i), dim=0) for x_i in x]
        return torch.stack(z)

    def _simple(self, x: Dict) -> Tensor:
        z = self.base_text_clip(**x).pooler_output
        return self.base_text_proj(z)

    def resize_embeddings_layer(self, num_tokens: int) -> None:
        """ Either augment or decrease the size of the embeddings layer of the
        underlying model.

        Args:
            num_tokens: Number of tokens in the tokenizer.

        """
        self.base.resize_token_embeddings(num_tokens)


class CustomVisualCLIP(nn.Module):

    def __init__(self, num_classes: int, finetune: bool = False):
        """
        Visual half of a CLIP model ready to fine-tune on classification
        and/or regression tasks.

        Args:
            num_classes: Labels to predict.
            finetune: If True, adjust only the last layer, otherwise train
            the whole model.

        """
        super(CustomVisualCLIP, self).__init__()
        clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.base_visual_clip = clip.vision_model
        self.base_visual_proj = clip.visual_projection
        if finetune:
            for layer in [self.base_visual_clip, self.base_visual_proj]:
                for param in layer.parameters():
                    param.requires_grad = False
        self.classifier = nn.Linear(self.base_visual_proj.out_features,
                                    num_classes)

    def forward(self, x: Dict) -> Tensor:
        """ Pass forward.

        Args:
            x: Dictionary of pixel values with shape (BS, F, C, H, W).

        Returns:
            prediction per video (BS, num_classes)
        """
        x['pixel_values'] = x['pixel_values'].squeeze()
        z = self.base_visual_clip(**x).pooler_output
        z = self.base_visual_proj(z)
        z = self.classifier(z)
        return nn.functional.log_softmax(z, dim=1)
