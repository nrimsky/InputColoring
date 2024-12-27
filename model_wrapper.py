import torch
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os
from enum import Enum
from utils import user_mask_function, assistant_mask_function
import torch.nn.functional as F

load_dotenv()

MODEL_LLAMA_3_CHAT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class Intervention(Enum):
    EMBEDDING_COLOR = 1
    RESID_ADD_PROJECT = 2
    STEER_AT_LAYER = 3


class InterventionSettings:
    def __init__(self, intervention: Intervention, **kwargs):
        self.intervention = intervention
        self.user_vector = kwargs["user_vector"]
        self.assistant_vector = kwargs["assistant_vector"]
        if intervention == Intervention.STEER_AT_LAYER:
            assert "layer" in kwargs, "need layer for STEER_AT_LAYER"
            self.layer = kwargs["layer"]


def register_hook_to_add_and_proj_to_token_repr(
    module: torch.nn.Module,
    mask_attr: str,
    vector: torch.Tensor,
    proj_vector: torch.Tensor | None,
):
    def hook_fn(module, input, output):
        acts = output
        if isinstance(acts, tuple):
            acts = acts[0]

        model_ref = module.model_ref
        mask = getattr(model_ref, mask_attr).unsqueeze(-1)  # b s 1
        expanded_vector = vector.unsqueeze(0).unsqueeze(0)  # 1 1 r
        acts += mask * expanded_vector  # b s r
        if proj_vector is not None:
            projections = torch.einsum("bsr,r->bs", acts, proj_vector).unsqueeze(
                -1
            )  # b s 1
            projections *= mask  # b s 1
            expanded_proj_vector = proj_vector.unsqueeze(0).unsqueeze(0)  # 1 1 r
            acts -= projections * expanded_proj_vector  # b s r
        if isinstance(output, tuple):
            return (acts, *output[1:])
        else:
            return acts

    handle = module.register_forward_hook(hook_fn)
    return handle


def main_hook_fn(module, input, output):
    module.user_mask = user_mask_function(input[0])
    module.assistant_mask = assistant_mask_function(input[0])


class ModelWrapper(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_LLAMA_3_CHAT,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.hooks = []

        # model refs
        self.model.model.embed_tokens.model_ref = self.model
        for block in self.model.model.layers:
            block.model_ref = self.model

    def reset_hooks(self):
        for handle in self.hooks:
            handle.remove()

    def set_intervention(self, settings: InterventionSettings):
        self.reset_hooks()

        self.hooks.append(self.model.register_forward_hook(main_hook_fn))

        user_vector = settings.user_vector.to(self.device).to(torch.bfloat16)
        assistant_vector = settings.assistant_vector.to(self.device).to(torch.bfloat16)

        # set up hooks based on intervention type
        if settings.intervention == Intervention.EMBEDDING_COLOR:
            self.hooks.append(
                register_hook_to_add_and_proj_to_token_repr(
                    self.model.model.embed_tokens,
                    "user_mask",
                    user_vector,
                    None,
                )
            )
            self.hooks.append(
                register_hook_to_add_and_proj_to_token_repr(
                    self.model.model.embed_tokens,
                    "assistant_mask",
                    assistant_vector,
                    None,
                )
            )
        elif settings.intervention == Intervention.RESID_ADD_PROJECT:
            for block in self.model.model.layers:
                self.hooks.append(
                    register_hook_to_add_and_proj_to_token_repr(
                        block,
                        "user_mask",
                        user_vector,
                        assistant_vector,
                    )
                )
                self.hooks.append(
                    register_hook_to_add_and_proj_to_token_repr(
                        block,
                        "assistant_mask",
                        assistant_vector,
                        user_vector,
                    )
                )
        elif settings.intervention == Intervention.STEER_AT_LAYER:
            for i, block in enumerate(self.model.model.layers):
                if i == settings.layer:
                    self.hooks.append(
                        register_hook_to_add_and_proj_to_token_repr(
                            block,
                            "user_mask",
                            user_vector,
                            None,
                        )
                    )
                    self.hooks.append(
                        register_hook_to_add_and_proj_to_token_repr(
                            block,
                            "assistant_mask",
                            assistant_vector,
                            None,
                        )
                    )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.model.generate(*args, **kwargs)

    @property
    def device(self):
        return self.model.device

    def get_per_token_nlls(self, tokens):
        tokens = tokens.to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens, return_dict=True)
            logits = outputs.logits[:, :-1, :]  # Remove last position from logits
            labels = tokens[:, 1:]  # Remove first position from labels
            nlls = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
            )
        return nlls.view(tokens.size(0), -1)  # Reshape to [batch_size, seq_len-1]
