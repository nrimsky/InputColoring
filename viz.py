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
from utils import format_conversation, assistant_mask_function, user_mask_function


def viz_mask(tokens, mask, tokenizer):
    tokens = tokens[0].cpu()
    mask = mask[0].cpu()
    html_parts = []
    for token, m in zip(tokens, mask):
        detok = tokenizer.decode(token.unsqueeze(0))
        html_parts.append(
            f"<span style='padding: 2px 4px; margin: 2px; display: inline-block; background-color: {'#FF0000' if m else '#FFFFFF'}; color: {'#FFFFFF' if m else '#000'}'>"
            f"{html.escape(detok)}"
            f"</span>"
        )
        if "\n" in detok:
            html_parts.append("<br>")
    display(HTML("".join(html_parts)))
