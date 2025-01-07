import torch
from torch.utils.data import Dataset
import json
import random
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
import os
from dataclasses import dataclass
from glob import glob
import gc
from model_wrapper import InterventionSettings, ModelWrapper, Intervention


@dataclass
class TrainingConfig:
    train_files: list[str]
    val_files: list[str]
    intervention_settings: InterventionSettings | None
    save_dir: str
    learning_rate: float = 1e-5
    num_epochs: int = 1
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    validation_samples: int = 30
    val_check_interval: float = 0.2
    trainable_layers: list[int] | None = None

    def to_dict(self):
        return {
            "train_files": self.train_files,
            "val_files": self.val_files,
            "intervention_settings": (
                self.intervention_settings.to_dict()
                if self.intervention_settings is not None
                else None
            ),
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "validation_samples": self.validation_samples,
            "val_check_interval": self.val_check_interval,
            "save_dir": self.save_dir,
            "trainable_layers": self.trainable_layers,
        }


class ConversationDataset(Dataset):
    def __init__(self, file_paths: list[str]):
        self.examples = []
        for file_path in file_paths:
            with open(file_path, "r") as f:
                file_data = json.load(f)
                self.examples.extend(file_data)
        random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "tokens": torch.tensor(example["tokens"]),
            "assistant_mask": torch.tensor(example["assistant_mask"]),
        }


def freeze_layers(model, trainable_layers: list[int]):
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Then unfreeze the specified layers
    for layer_idx in trainable_layers:
        for param in model.model.layers[layer_idx].parameters():
            param.requires_grad = True


def compute_masked_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, reduction: str
) -> torch.Tensor:
    # logits: [batch_size, seq_len-1, vocab_size]
    # labels: [batch_size, seq_len-1]
    # mask: [batch_size, seq_len-1]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
    )
    loss = loss.view(labels.size(0), -1)
    if reduction == "mean":
        masked_loss = (loss * mask).sum() / mask.sum()
    elif reduction == "sum":
        masked_loss = (loss * mask).sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
    return masked_loss


def evaluate_model(
    model: ModelWrapper, val_files: list[str], n_samples: int
) -> dict[str, float]:
    model.eval()
    results = {}

    for val_file in val_files:
        with open(val_file, "r") as f:
            val_data = json.load(f)

        val_samples = random.sample(val_data, min(n_samples, len(val_data)))
        total_loss = 0.0

        for sample in val_samples:
            tokens = torch.tensor(sample["tokens"]).to(model.device)
            mask = torch.tensor(sample["assistant_mask"]).to(model.device)

            with torch.no_grad():
                outputs = model(tokens, return_dict=True)
                logits = outputs.logits[:, :-1, :]
                labels = tokens[:, 1:]
                loss = compute_masked_loss(
                    logits, labels, mask[:, 1:], reduction="mean"
                )
                total_loss += loss.item()

        avg_loss = total_loss / len(val_samples)
        results[os.path.basename(val_file)] = avg_loss

    model.train()
    return results


def print_and_log(string, fh):
    print(string)
    fh.write(string + "\n")


def train_model(model: ModelWrapper, config: TrainingConfig):
    logfile = f"{config.save_dir}/training_log.txt"
    os.makedirs(config.save_dir, exist_ok=True)
    # Clear log file if it already exists
    if os.path.exists(logfile):
        os.remove(logfile)
    # Save training config
    with open(f"{config.save_dir}/training_config.json", "w") as f:
        json.dump(config.to_dict(), f)
    train_dataset = ConversationDataset(config.train_files)
    print(f"Training on {len(train_dataset)} samples")

    logfilehandle = open(logfile, "w")

    if config.use_lora:
        target_modules = []
        if config.trainable_layers is not None:
            for layer_idx in config.trainable_layers:
                target_modules.extend(
                    [
                        f"model.layers.{layer_idx}.self_attn.q_proj",
                        f"model.layers.{layer_idx}.self_attn.v_proj",
                    ]
                )
        else:
            target_modules = ["q_proj", "v_proj"]  # Default behavior

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()
    else:
        # For full finetuning, freeze/unfreeze specified layers
        if config.trainable_layers is not None:
            freeze_layers(model, config.trainable_layers)

    # get validation stats before intervention + training
    val_results = evaluate_model(model, config.val_files, config.validation_samples)
    print_and_log(
        "Validation results before training and without intervention:", logfilehandle
    )
    for file_name, val_loss in val_results.items():
        print_and_log(f"{file_name}: {val_loss:.4f}", logfilehandle)

    # attach intervention hooks
    if config.intervention_settings is not None:
        model.set_intervention(config.intervention_settings)

    # get validation stats after intervention + before training
    val_results = evaluate_model(model, config.val_files, config.validation_samples)
    print_and_log(
        "Validation results before training but with intervention:", logfilehandle
    )
    for file_name, val_loss in val_results.items():
        print_and_log(f"{file_name}: {val_loss:.4f}", logfilehandle)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    total_steps = len(train_dataset) * config.num_epochs
    val_check_steps = int(total_steps * config.val_check_interval)

    global_step = 0
    train_loss = 0.0
    n = 0
    model.train()

    for epoch in range(config.num_epochs):
        print_and_log(f"\nEpoch {epoch + 1}/{config.num_epochs}", logfilehandle)
        progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}")
        for row in progress_bar:
            tokens = row["tokens"].to(model.device)
            mask = row["assistant_mask"].to(model.device)
            outputs = model(tokens, return_dict=True)
            logits = outputs.logits[:, :-1, :]  # Remove last position
            labels = tokens[:, 1:]  # Remove first position
            loss = compute_masked_loss(logits, labels, mask[:, 1:], reduction="mean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            train_loss += loss.item()
            global_step += 1
            n += 1
            if global_step % val_check_steps == 0:
                val_results = evaluate_model(
                    model, config.val_files, config.validation_samples
                )
                print_and_log(
                    f"Avg train loss over last {n} steps: {train_loss/n:.4f}",
                    logfilehandle,
                )
                print_and_log("Validation results:", logfilehandle)
                for file_name, val_loss in val_results.items():
                    print_and_log(f"{file_name}: {val_loss:.4f}", logfilehandle)
                train_loss = 0.0
                n = 0

    logfilehandle.close()

    if config.use_lora:
        model.model.save_pretrained(config.save_dir)
    else:
        torch.save(model.model.state_dict(), f"{config.save_dir}/pytorch_model.bin")


def model_wrapper_fn(fn):
    def wrapper():
        model = None
        try:
            model = ModelWrapper()
            fn(model)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return wrapper


@model_wrapper_fn
def embedding_color(model: ModelWrapper):
    user_token_embedding = model.user_token_embedding.clone()
    assistant_token_embedding = model.assistant_token_embedding.clone()
    intervention = InterventionSettings(
        intervention=Intervention.EMBEDDING_COLOR,
        user_vector=user_token_embedding,
        assistant_vector=assistant_token_embedding,
        scale_factor=0.8,
    )
    config = TrainingConfig(
        train_files=glob("processed_data/train/*.json"),
        val_files=glob("processed_data/test/*.json"),
        intervention_settings=intervention,
        use_lora=True,
        save_dir="saved_models/embedding_color",
    )
    train_model(model, config)


@model_wrapper_fn
def embedding_color_train_5_to_10(model: ModelWrapper):
    user_token_embedding = model.user_token_embedding.clone()
    assistant_token_embedding = model.assistant_token_embedding.clone()
    intervention = InterventionSettings(
        intervention=Intervention.EMBEDDING_COLOR,
        user_vector=user_token_embedding,
        assistant_vector=assistant_token_embedding,
        scale_factor=0.8,
    )
    config = TrainingConfig(
        train_files=glob("processed_data/train/*.json"),
        val_files=glob("processed_data/test/*.json"),
        intervention_settings=intervention,
        use_lora=True,
        save_dir="saved_models/embedding_color_train_5_to_10",
        trainable_layers=list(range(5, 10)),
    )
    train_model(model, config)


@model_wrapper_fn
def steer_5_train_10_through_15(model: ModelWrapper):
    user_token_embedding = model.user_token_embedding.clone()
    assistant_token_embedding = model.assistant_token_embedding.clone()
    intervention = InterventionSettings(
        intervention=Intervention.STEER_AT_LAYER,
        user_vector=user_token_embedding,
        assistant_vector=assistant_token_embedding,
        scale_factor=0.5,
        layer=5,
    )
    config = TrainingConfig(
        train_files=glob("processed_data/train/*.json"),
        val_files=glob("processed_data/test/*.json"),
        intervention_settings=intervention,
        use_lora=True,
        save_dir="saved_models/steer_5_train_10_through_15",
        trainable_layers=list(range(10, 15)),
    )
    train_model(model, config)


@model_wrapper_fn
def resid_color(model: ModelWrapper):
    user_token_embedding = model.user_token_embedding.clone()
    assistant_token_embedding = model.assistant_token_embedding.clone()
    intervention = InterventionSettings(
        intervention=Intervention.RESID_ADD_PROJECT,
        user_vector=user_token_embedding,
        assistant_vector=assistant_token_embedding,
        scale_factor=0.2,
        skip_last_layer=True,
    )
    config = TrainingConfig(
        train_files=glob("processed_data/train/*.json"),
        val_files=glob("processed_data/test/*.json"),
        intervention_settings=intervention,
        use_lora=True,
        save_dir="saved_models/resid_color",
    )
    train_model(model, config)


@model_wrapper_fn
def control(model: ModelWrapper):
    config = TrainingConfig(
        train_files=glob("processed_data/train/*.json"),
        val_files=glob("processed_data/test/*.json"),
        intervention_settings=None,
        use_lora=True,
        save_dir="saved_models/control",
    )
    train_model(model, config)


if __name__ == "__main__":
    # embedding_color()
    # embedding_color_train_5_to_10()
    steer_5_train_10_through_15()
    resid_color()
    control()
