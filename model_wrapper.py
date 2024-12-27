import torch
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os
from enum import Enum
from utils import SPECIAL_ZONE_START, user_mask_function, assistant_mask_function, START_HEADER_ID, END_HEADER_ID, ASSISTANT_ID, USER_ID
import torch.nn.functional as F

load_dotenv()

MODEL_LLAMA_3_CHAT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class Intervention(Enum):
    EMBEDDING_COLOR = "embedding_color"
    RESID_ADD_PROJECT = "resid_add_project"
    STEER_AT_LAYER = "steer_at_layer"


class InterventionSettings:
    def __init__(self, intervention: Intervention, **kwargs):
        self.intervention = intervention
        self.user_vector = kwargs["user_vector"]
        self.assistant_vector = kwargs["assistant_vector"]
        if intervention == Intervention.STEER_AT_LAYER:
            assert "layer" in kwargs, "need layer for STEER_AT_LAYER"
            self.layer = kwargs["layer"]

    def __str__(self):
        int_name = self.intervention.value
        if hasattr(self, "layer"):
            int_name += f"_{self.layer}"
        return int_name
    
    def __repr__(self):
        return str(self)

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
        # print(mask_attr, mask.sum().item())
        expanded_vector = vector.unsqueeze(0).unsqueeze(0)  # 1 1 r
        acts += mask * expanded_vector  # b s r
        if proj_vector is not None:
            projections = torch.einsum("bsr,r->bs", acts, proj_vector).unsqueeze(
                -1
            )  # b s 1
            projections[projections < 0] = 0
            projections *= mask  # b s 1
            expanded_proj_vector = proj_vector.unsqueeze(0).unsqueeze(0)  # 1 1 r
            acts -= projections * expanded_proj_vector  # b s r
        if isinstance(output, tuple):
            return (acts, *output[1:])
        else:
            return acts

    handle = module.register_forward_hook(hook_fn)
    return handle


def main_hook_fn(module, inputs):
    # Typically for embed_tokens, inputs[0] should be the input_ids tensor.
    # But depending on HF version, it could be a dictionary.
    if isinstance(inputs[0], dict):
        # If it's a dict, we want the "input_ids" key
        input_ids = inputs[0].get("input_ids", None)
    else:
        # If it's not a dict, we assume it's the raw tensor
        input_ids = inputs[0]

    module.model_ref.user_mask = user_mask_function(input_ids)
    module.model_ref.assistant_mask = assistant_mask_function(input_ids)

    # Return the same set of inputs so the forward continues
    return inputs



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
        self.hooks = []

    def set_intervention(self, settings: InterventionSettings):
        self.reset_hooks()

        self.hooks.append(
            self.model.model.embed_tokens.register_forward_pre_hook(main_hook_fn)
        )


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
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def sample(self, tokens: torch.Tensor, max_n_tokens: int) -> torch.Tensor:
        attention_mask = torch.ones_like(tokens)
        return self.model.generate(
            tokens,
            max_length=max_n_tokens,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    @property
    def device(self):
        return self.model.device

    def get_per_token_nlls(self, tokens: torch.Tensor):
        tokens = tokens.to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens, return_dict=True)
            logits = outputs.logits[:, :-1, :]  # Remove last position from logits
            labels = tokens[:, 1:]  # Remove first position from labels
            nlls = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
            )
        return nlls.view(tokens.size(0), -1)  # Reshape to [batch_size, seq_len-1]
    
    @property
    def assistant_token_embedding(self):
        return self.model.model.embed_tokens.weight[ASSISTANT_ID].detach()
    
    @property
    def user_token_embedding(self):
        return self.model.model.embed_tokens.weight[USER_ID].detach()
    
    def get_role_nlls(self, tokens: torch.Tensor):
        nlls = self.get_per_token_nlls(tokens)
        tokens = tokens[0].cpu()[1:]  # remove SOS token which doesn't have an associated NLL
        results = []
        current_role = None
        current_nll_sum = 0
        role_started = False 
        current_tokens = []

        for i, (token, nll) in enumerate(zip(tokens, nlls[0])):
            if token == START_HEADER_ID:
                if current_role is not None:
                    results.append({
                        "role": current_role,
                        "nll": current_nll_sum.item(),
                        "token_count": len(current_tokens),
                        "tokens": current_tokens
                    })
                current_role = int(tokens[i+1])
                current_nll_sum = 0
                role_started = False
                current_tokens = []
            elif token == END_HEADER_ID:
                role_started = True
            elif token >= SPECIAL_ZONE_START:
                continue
            elif role_started:
                current_nll_sum += nll
                current_tokens.append(int(token))
        if current_role is not None:
            results.append({
                "role": current_role,
                "nll": current_nll_sum.item(),
                "token_count": len(current_tokens),
                "tokens": current_tokens
            })
        return results
        
