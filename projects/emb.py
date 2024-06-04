"""
Functions for loading and using various embedding models.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
import transformers as tfs
from transformers import AutoTokenizer, AutoModel


E5_MAX_LENGTH = 512


def load_e5(model_name: str, cache_dir_path: str, device: str) -> Tuple[tfs.BertTokenizerFast, tfs.BertModel]:
    """Load E5 tokenizer and model."""

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir_path)

    # this ends up being a BertModel
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map=device,
        cache_dir=cache_dir_path)

    return tokenizer, model


def e5_embeddings(
        tokenizer: tfs.BertTokenizerFast,
        model: tfs.BertModel,
        text_batch: List[str]
        ) -> torch.Tensor:
    """Calculate a batch of embeddings with an E5 model."""

    device = model.device

    # Tokenize the input texts
    # TODO: double check max length here
    batch_dict = tokenizer(text_batch, max_length=E5_MAX_LENGTH, padding=True, truncation=True, return_tensors='pt')
    batch_dict = batch_dict.to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def average_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
    """Average pooling using an attention mask."""

    # this is how things were written in HuggingFace examples
    # for example: https://huggingface.co/intfloat/e5-base-v2
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # I personally think this is clearer and I think it should be equivalent
    # last_hidden = last_hidden_states * attention_mask[..., None]
    # return torch.sum(last_hidden, dim=1) / torch.sum(attention_mask, dim=1)[..., None]
