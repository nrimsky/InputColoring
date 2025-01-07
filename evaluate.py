import torch
import os
from dotenv import load_dotenv
import json
from model_wrapper import ModelWrapper, InterventionSettings
from collections import defaultdict
from tqdm import tqdm
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

DATASETS = [
    "icl_sequences",
    "parity_sequences",
    "regular_conversations",
    "msjs_jailbreak",
    "msjs_recovery",
    "msjs_mean_recovery",
    "msjs_mean_jailbreak"
]


def get_msj_nlls(
    model: ModelWrapper,
    dataset_path: str,
    intervention: InterventionSettings | None = None,
    model_name: str | None = None,
):
    # construct result save path
    result_save_path = dataset_path.replace("processed_data", "results")
    if intervention is not None:
        if model_name is not None:
            result_save_path = result_save_path.replace(
                ".json", f"_{model_name}_{intervention}.json"
            )
        else:
            result_save_path = result_save_path.replace(
                ".json", f"_{intervention}.json"
            )
    elif model_name is not None:
        result_save_path = result_save_path.replace(".json", f"_{model_name}.json")
    # check if result already exists
    if os.path.exists(result_save_path):
        with open(result_save_path) as f:
            print(f"Found existing results at {result_save_path}")
            return result_save_path
    # run the model on the dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    result = defaultdict(list)
    for row in tqdm(dataset):
        tokens = torch.tensor(row["tokens"]).to(model.device)
        nlls = model.get_role_nlls(tokens)
        result[row["n_shots"]].append(nlls)
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    with open(result_save_path, "w") as f:
        json.dump(result, f)
    return result_save_path


def get_all_dataset_nlls_for_model(
    model_path: str, use_lora: bool = True, inference_only: bool = False
):
    with open(os.path.join(model_path, "training_config.json")) as f:
        config = json.load(f)
        intervention_config = config["intervention_settings"]
        if intervention_config is not None:
            intervention = InterventionSettings.from_dict(intervention_config)
        else:
            intervention = None
    model_name = os.path.split(model_path)[-1].split(".")[0]
    model = ModelWrapper()
    if not inference_only:
        model.load_weights(model_path, use_lora=use_lora)
    if intervention is not None:
        model.set_intervention(intervention)
    result_paths = {}
    for d in DATASETS:
        dataset_path = os.path.join("processed_data", "test", d)+".json"
        sp = get_msj_nlls(
            model,
            dataset_path,
            intervention,
            model_name if not inference_only else None,
        )
        result_paths[d] = sp
    return result_paths

def plot_nlls(results, legend, filename, figsize=(7, 4), title = None):
    fig, ax = plt.subplots(figsize=figsize)
    for r, l in zip(results, legend):
        if isinstance(r, str):
            with open(r) as f:
                r = json.load(f)
        data = sorted([
            (int(k), sum([r[-1]['nll'] for r in v])/len(v)) for k, v in r.items()
        ])
        shots, nlls = zip(*data)
        shots = np.array(shots)
        ax.plot(shots, nlls, 'o-', markersize=5, label=l)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_xlabel('Number of shots')
    ax.set_ylabel('NLL of final assistant response')    
    ax.legend()
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")

def run_all_models():
    model_paths = glob.glob("saved_models/*")
    results = defaultdict(list)
    for model_path in model_paths:
        p = get_all_dataset_nlls_for_model(model_path)
        print("Finished trained model evals for", model_path)
        print(p)
        for eval_name, eval_path in p.items():
            results[eval_name].append(model_path, eval_path)
    for eval_name, eval_results in results.items():
        model_names, nll_paths = zip(*eval_results)
        model_names = [m.split("/")[-1] for m in model_names]
        plot_nlls(nll_paths, model_names, eval_name, title = eval_name)

def run_inference_time_eval(model_paths):
    for model_path in model_paths:
        get_all_dataset_nlls_for_model(model_path, inference_only=True)
        print("Finished inference only evals for", model_path)



if __name__ == "__main__":
    run_all_models()
