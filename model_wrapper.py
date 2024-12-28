import torch
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os
from enum import Enum
from utils import SPECIAL_ZONE_START, user_mask_function, assistant_mask_function, START_HEADER_ID, END_HEADER_ID, ASSISTANT_ID, USER_ID
import torch.nn.functional as F
import random
from peft import PeftModelForCausalLM
from transformers.models.llama import LlamaForCausalLM

LOG_DEBUG = False

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
        self.norm_factor = kwargs.get("norm_factor", None)
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

    def to_dict(self):
        return {
            "intervention": self.intervention.value,
            "user_vector": self.user_vector.tolist(),
            "assistant_vector": self.assistant_vector.tolist(),
            "norm_factor": self.norm_factor,
            "layer": getattr(self, "layer", None),
        }

def register_hook_to_add_and_proj_to_token_repr(
    module: torch.nn.Module,
    mask_attr: str,
    vector: torch.Tensor,
    proj_vector: torch.Tensor | None,
    norm_factor: float | None = None,
):
    def hook_fn(module, input, output):
        # Extract main activations from output
        acts = output[0] if isinstance(output, tuple) else output

        data_ref = module.data_ref
        mask = getattr(data_ref, mask_attr).unsqueeze(-1)  # b s 1

        batch_size, seq_len, _ = acts.shape

        expanded_vector = vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        if norm_factor is not None:
            act_norms = torch.norm(acts.detach(), p=2, dim=-1).unsqueeze(-1).expand_as(mask)
            vector_norm = torch.norm(vector, p=2)
            multipliers = act_norms / vector_norm
            expanded_vector = expanded_vector * (norm_factor * multipliers)
        
        acts += mask * expanded_vector
        
        # Optionally remove projection
        if proj_vector is not None:
            projections = torch.einsum("bsr,r->bs", acts, proj_vector).unsqueeze(-1)  # b s 1
            projections *= mask
            expanded_proj_vector = proj_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            acts -= projections * expanded_proj_vector
        
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

    module.data_ref.user_mask = user_mask_function(input_ids)
    module.data_ref.assistant_mask = assistant_mask_function(input_ids)

    if LOG_DEBUG:
        if random.random() < 0.01:
            print(f"running main hook on module: {module.__class__.__name__}, user_mask_sum={module.data_ref.user_mask.sum().item()}, assistant_mask_sum={module.data_ref.assistant_mask.sum().item()}")

    # Return the same set of inputs so the forward continues
    return inputs


class InterventionData:
    def __init__(self):
        self.user_mask = None
        self.assistant_mask = None

class ModelWrapper(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_LLAMA_3_CHAT,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.intervention_data = InterventionData()
        self.hooks = []
        self.configure_refs()

    @property
    def embedding(self):
        if isinstance(self.model, PeftModelForCausalLM):
            return self.model.base_model.model.model.embed_tokens
        elif isinstance(self.model, LlamaForCausalLM):
            return self.model.model.embed_tokens
        
    @property
    def blocks(self):
        if isinstance(self.model, PeftModelForCausalLM):
            return self.model.base_model.model.model.layers
        elif isinstance(self.model, LlamaForCausalLM):
            return self.model.model.layers

    def configure_refs(self):
        self.embedding.data_ref = self.intervention_data
        for block in self.blocks:
            block.data_ref = self.intervention_data

    def reset_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def set_intervention(self, settings: InterventionSettings):
        self.reset_hooks()
        self.configure_refs()
        embedding = self.embedding
        blocks = self.blocks
        self.hooks.append(
            embedding.register_forward_pre_hook(main_hook_fn)
        )

        user_vector = settings.user_vector.to(self.device).to(torch.bfloat16)
        assistant_vector = settings.assistant_vector.to(self.device).to(torch.bfloat16)

        # set up hooks based on intervention type
        if settings.intervention == Intervention.EMBEDDING_COLOR:
            self.hooks.append(
                register_hook_to_add_and_proj_to_token_repr(
                    embedding,
                    "user_mask",
                    user_vector,
                    None,
                    settings.norm_factor
                )
            )
            self.hooks.append(
                register_hook_to_add_and_proj_to_token_repr(
                    embedding,
                    "assistant_mask",
                    assistant_vector,
                    None,
                    settings.norm_factor
                )
            )
        elif settings.intervention == Intervention.RESID_ADD_PROJECT:
            for block in blocks:
                self.hooks.append(
                    register_hook_to_add_and_proj_to_token_repr(
                        block,
                        "user_mask",
                        user_vector,
                        assistant_vector,
                        settings.norm_factor
                    )
                )
                self.hooks.append(
                    register_hook_to_add_and_proj_to_token_repr(
                        block,
                        "assistant_mask",
                        assistant_vector,
                        user_vector,
                        settings.norm_factor
                    )
                )
        elif settings.intervention == Intervention.STEER_AT_LAYER:
            for i, block in enumerate(blocks):
                if i == settings.layer:
                    self.hooks.append(
                        register_hook_to_add_and_proj_to_token_repr(
                            block,
                            "user_mask",
                            user_vector,
                            None,
                            settings.norm_factor
                        )
                    )
                    self.hooks.append(
                        register_hook_to_add_and_proj_to_token_repr(
                            block,
                            "assistant_mask",
                            assistant_vector,
                            None,
                            settings.norm_factor
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
        return self.embedding.weight[ASSISTANT_ID].data.detach().clone()
    
    @property
    def user_token_embedding(self):
        return self.embedding.weight[USER_ID].data.detach().clone()
    
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
    
    def load_weights(self, model_dir: str, use_lora: bool):
        """
        Load weights from a saved model directory.
        
        Args:
            model_dir: Directory containing saved weights, should match the intervention-based
                    directory structure used in training (e.g. 'saved_models/embedding_color')
            use_lora: Whether to load LORA weights (should match training configuration)
        """
        if use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            # First ensure model is set up for LORA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,  # We're loading for inference
                r=8,  # These values don't matter for inference as they'll be overwritten
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                target_modules=["q_proj", "v_proj"]
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.load_adapter(model_dir, adapter_name="default")
        else:
            # Load full model weights
            weights_path = os.path.join(model_dir, "pytorch_model.bin")
            if not os.path.exists(weights_path):
                raise ValueError(f"No weights found at {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.configure_refs()
        print("loaded weights from", model_dir, "using LORA:", use_lora)
        print("you probably want to call model.set_intervention(...) to set up any intervention hooks")
