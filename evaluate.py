import torch
import os
from dotenv import load_dotenv
import json
from model_wrapper import ModelWrapper, InterventionSettings
from collections import defaultdict
from tqdm import tqdm
import glob

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


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
            print(f"Loading result from {result_save_path}")
            return json.load(f)
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
    data_path: str, model_path: str, use_lora: bool = True, inference_only: bool = False
):
    datasets = glob.glob(os.path.join(data_path, "*.json"))
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
    for dataset_path in datasets:
        sp = get_msj_nlls(
            model,
            dataset_path,
            intervention,
            model_name if not inference_only else None,
        )
        result_paths[dataset_path] = sp
    return result_paths


def run_all_models():
    model_paths = glob.glob("saved_models")
    data_path = "processed_data/test"
    for model_path in model_paths:
        p = get_all_dataset_nlls_for_model(data_path, model_path, inference_only=True)
        print("Finished inference only evals for", model_path)
        print(p)
        get_all_dataset_nlls_for_model(data_path, model_path)
        print("Finished trained model evals for", model_path)
        print(p)


if __name__ == "__main__":
    run_all_models()
