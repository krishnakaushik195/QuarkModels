import os
import torch
from model_architecture import BigramLanguageModel
from model_loader import download_and_load_model

HF_REPO_ID = "QuarkModels/quarks_baisc_chat"
HF_TOKEN = os.getenv("HF_TOKEN")  # read token from env variable

device = "cuda" if torch.cuda.is_available() else "cpu"

model = download_and_load_model(HF_REPO_ID, HF_TOKEN, BigramLanguageModel)
model.to(device)

prompt = "hello"
# encode your prompt, generate, decode etc.
