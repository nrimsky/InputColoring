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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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

def visualize_token_nlls(tokens, nlls, tokenizer, cmap='RdYlBu_r'):
    cmap = plt.get_cmap(cmap)
    # Normalize NLL values to [0,1] for color mapping
    vmin, vmax = np.percentile(nlls, [5, 95])  # Use percentiles to handle outliers
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Generate HTML for the visualization
    html_parts = []
    # Add legend
    html_parts.append("""
    <div style='margin-bottom: 10px'>
        <span style='font-size: 0.9em'>NLL Range: </span>
        <span style='padding: 2px 8px; background-color: {}; color: white'>Low {:.2f}</span>
        <span style='padding: 2px 8px; background-color: {}'>Med {:.2f}</span>
        <span style='padding: 2px 8px; background-color: {}; color: white'>High {:.2f}</span>
    </div>
    """.format(
        mcolors.rgb2hex(cmap(0.0)),
        vmin,
        mcolors.rgb2hex(cmap(0.5)),
        (vmax + vmin) / 2,
        mcolors.rgb2hex(cmap(1.0)),
        vmax
    ))
    # Add first token (which has no NLL)
    html_parts.append(f"<span style='padding: 2px 4px; margin: 0 1px; border: 1px dashed #ccc'>{tokens[0]}</span>")
    # Add remaining tokens with color coding
    for token, nll in zip(tokens[1:], nlls, strict=True):
        # Get color for this NLL value
        color = mcolors.rgb2hex(cmap(norm(nll)))
        # Determine text color (white for dark backgrounds)
        rgb = mcolors.hex2color(color)
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = 'white' if brightness < 0.5 else 'black'
        # Add token with styling
        detok = tokenizer.decode(token.unsqueeze(0))
        html_parts.append(
            f"<span style='padding: 2px 4px; margin: 2px; display: inline-block; background-color: {color}; "
            f"color: {text_color}' title='NLL: {nll:.3f}'>{html.escape(detok)}</span>"
        )
    # Display the HTML
    display(HTML(''.join(html_parts)))