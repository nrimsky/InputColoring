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

from model_wrapper import InterventionSettings, ModelWrapper, Intervention

@dataclass
class TrainingConfig:
    train_files: list[str]
    val_files: list[str]
    intervention_settings: InterventionSettings|None
    learning_rate: float = 1e-5
    num_epochs: int = 1
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    validation_samples: int = 30
    val_check_interval: float = 0.1

class ConversationDataset(Dataset):
    def __init__(self, file_paths: list[str]):
        self.examples = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                self.examples.extend(file_data)
        random.shuffle(self.examples)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'tokens': torch.tensor(example['tokens']),
            'assistant_mask': torch.tensor(example['assistant_mask'])
        }

def compute_masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: [batch_size, seq_len-1, vocab_size]
    # labels: [batch_size, seq_len-1]
    # mask: [batch_size, seq_len-1]
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                          labels.view(-1), 
                          reduction='none')
    loss = loss.view(labels.size(0), -1)
    masked_loss = (loss * mask).sum() / mask.sum()
    return masked_loss

def evaluate_model(model: ModelWrapper, val_files: list[str], n_samples: int) -> dict[str, float]:
    model.eval()
    results = {}
    
    for val_file in val_files:
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        
        val_samples = random.sample(val_data, min(n_samples, len(val_data)))
        total_loss = 0.0
        
        for sample in val_samples:
            tokens = torch.tensor(sample['tokens']).to(model.device)
            mask = torch.tensor(sample['assistant_mask']).to(model.device)
            
            with torch.no_grad():
                outputs = model(tokens, return_dict=True)
                logits = outputs.logits[:, :-1, :]
                labels = tokens[:, 1:]
                loss = compute_masked_loss(logits, labels, mask[:, 1:])
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_samples)
        results[os.path.basename(val_file)] = avg_loss
    
    model.train()
    return results

def print_and_log(string, filename):
    print(string)
    with open(filename, 'a') as f:
        f.write(string + "\n")

def train_model(model: ModelWrapper, config: TrainingConfig):
    save_dir = f"saved_models/{str(config.intervention_settings)}"
    logfile = f"{save_dir}/training_log.txt"
    os.makedirs(save_dir, exist_ok=True)
    # delete log file if it already exists
    if os.path.exists(logfile):
        os.remove(logfile)
    train_dataset = ConversationDataset(config.train_files)
    print(f"Training on {len(train_dataset)} samples")

    if config.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"]
        )
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()

    # get validation stats before intervention + training
    val_results = evaluate_model(
        model,
        config.val_files,
        config.validation_samples
    )
    print_and_log("Validation results before training and without intervention:", logfile)
    for file_name, val_loss in val_results.items():
        print_and_log(f"{file_name}: {val_loss:.4f}", logfile)

    # attach intervention hooks
    if config.intervention_settings is not None:
        model.set_intervention(config.intervention_settings)

    # get validation stats after intervention + before training
    val_results = evaluate_model(
        model,
        config.val_files,
        config.validation_samples
    )
    print_and_log("Validation results before training but with intervention:", logfile)
    for file_name, val_loss in val_results.items():
        print_and_log(f"{file_name}: {val_loss:.4f}", logfile)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    total_steps = len(train_dataset) * config.num_epochs
    val_check_steps = int(total_steps * config.val_check_interval)
    
    global_step = 0
    train_loss = 0.0
    n = 0
    model.train()
    
    for epoch in range(config.num_epochs):
        print_and_log(f"\nEpoch {epoch + 1}/{config.num_epochs}", logfile)
        progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}")
        for row in progress_bar:
            tokens = row['tokens'].to(model.device)
            mask = row['assistant_mask'].to(model.device)
            outputs = model(tokens, return_dict=True)
            logits = outputs.logits[:, :-1, :]  # Remove last position
            labels = tokens[:, 1:]  # Remove first position
            loss = compute_masked_loss(logits, labels, mask[:, 1:])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_loss += loss.item()
            global_step += 1
            n += 1
            if global_step % val_check_steps == 0:
                val_results = evaluate_model(
                    model,
                    config.val_files,
                    config.validation_samples
                )
                print_and_log(f"Avg train loss over last {n} steps: {train_loss/n:.4f}", logfile)
                print_and_log("Validation results:", logfile)
                for file_name, val_loss in val_results.items():
                    print_and_log(f"{file_name}: {val_loss:.4f}", logfile)
                train_loss = 0.0
                n = 0

    if config.use_lora:
        model.model.save_pretrained(save_dir)
    else:
        torch.save(model.model.state_dict(), f"{save_dir}/pytorch_model.bin")
        
    

def exp1():
    model = ModelWrapper()
    user_token_embedding = model.user_token_embedding.clone()
    assistant_token_embedding = model.assistant_token_embedding.clone()
    interventions = [
        # InterventionSettings(
        #     intervention=Intervention.RESID_ADD_PROJECT,
        #     user_vector=user_token_embedding,
        #     assistant_vector=assistant_token_embedding,
        # ),
        InterventionSettings(
            intervention=Intervention.EMBEDDING_COLOR,
            user_vector=user_token_embedding,
            assistant_vector=assistant_token_embedding,
        ),
        # InterventionSettings(
        #     intervention=Intervention.STEER_AT_LAYER,
        #     user_vector=user_token_embedding,
        #     assistant_vector=assistant_token_embedding,
        #     layer=10
        # ),
        # None,
    ]
    for i in interventions:
        model = ModelWrapper()
        config = TrainingConfig(
            train_files=glob("processed_data/train/*.json"),
            val_files=glob("processed_data/test/*.json"),
            intervention_settings=i,
            use_lora=True 
        )
        train_model(model, config)


if __name__ == "__main__":
    exp1()