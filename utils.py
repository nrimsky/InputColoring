import json
import os
import random
from dotenv import load_dotenv
from transformers import AutoTokenizer
from IPython.display import display, HTML
from tqdm import tqdm
import html
import anthropic
import torch
from vars import assistant_fake_tags, user_fake_tags, format_functions
from datasets import load_dataset

USER_ID = 882
ASSISTANT_ID = 78191
START_HEADER_ID = 128006
END_HEADER_ID = 128007
SPECIAL_ZONE_START = 128000


def remove_system_prompt(text):
    prefix = text.split("<|eot_id|>")[0] + "<|eot_id|>"
    return text.removeprefix(prefix)


def format_conversation(conversation, tokenizer, add_generation_prompt=False):
    formatted_conversation = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=add_generation_prompt
    ).replace("<|begin_of_text|>", "")
    return remove_system_prompt(formatted_conversation)


def get_mask(
    tokens,
    target_role_token,
    start_header_token=START_HEADER_ID,
    end_header_token=END_HEADER_ID,
    apply_in_generate_mode=False,
):
    assert len(tokens.shape) == 2
    batch_size, seq_len = tokens.shape
    if seq_len == 1:
        # we are generating token by token
        if apply_in_generate_mode:
            return torch.ones_like(tokens)
        else:
            return torch.zeros_like(tokens)
    mask = torch.zeros_like(tokens)
    for i in range(batch_size):
        seq = tokens[i]
        role_started = False
        for j, t in enumerate(seq):
            if t == start_header_token:
                role_started = False
                role_token = seq[j + 1]
            elif t == end_header_token and role_token == target_role_token:
                role_started = True
                mask[i, j - 2] = 1
                mask[i, j - 1] = 1
                mask[i, j] = 1
            elif role_started:
                mask[i, j] = 1
    return mask


assistant_mask_function = lambda tokens: get_mask(
    tokens, ASSISTANT_ID, apply_in_generate_mode=True
)
user_mask_function = lambda tokens: get_mask(
    tokens, USER_ID, apply_in_generate_mode=False
)