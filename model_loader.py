import torch
import json
from huggingface_hub import snapshot_download

def download_and_load_model(repo_id, token, model_class):
    local_dir = snapshot_download(repo_id=repo_id, token=token)

    # Load config
    with open(f"{local_dir}/config.json", "r") as f:
        config = json.load(f)

    # Initialize model from config
    model = model_class(config)
    # Load weights
    model.load_state_dict(torch.load(f"{local_dir}/pytorch_model.bin", map_location="cpu"))
    model.eval()
    return model
