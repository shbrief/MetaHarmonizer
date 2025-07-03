from sentence_transformers import SentenceTransformer
import torch
import yaml
from functools import lru_cache
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

DEFAULT_YAML_PATH = os.getenv("METHOD_MODEL_YAML",
                              "src/models/method_model.yaml")
CACHE_ROOT = "model_cache"


def load_method_model_dict(yaml_path: str = DEFAULT_YAML_PATH) -> dict:
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)


def get_model(method: str,
              yaml_path: str = DEFAULT_YAML_PATH) -> SentenceTransformer:
    method_model_dict = load_method_model_dict(yaml_path)
    if method not in method_model_dict:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of {list(method_model_dict.keys())}"
        )
    repo_id = method_model_dict[method]
    local_dir = os.path.join(CACHE_ROOT, method)

    if not os.path.isdir(local_dir):
        print(
            f"Downloading model for method '{method}' from Hugging Face Hub..."
        )
        snapshot_download(repo_id=repo_id,
                          local_dir=local_dir,
                          cache_dir=CACHE_ROOT,
                          local_files_only=False,
                          repo_type="model",
                          revision=None,
                          allow_patterns=None,
                          ignore_patterns=None)

        downloaded = os.listdir(CACHE_ROOT)
        hf_dir = max(
            downloaded,
            key=lambda d: os.path.getmtime(os.path.join(CACHE_ROOT, d)))
        os.replace(os.path.join(CACHE_ROOT, hf_dir), local_dir)
        print(f"Successfully downloaded {repo_id} to {local_dir}")

    model = SentenceTransformer(
        local_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True)

    model._is_normalized = getattr(model, "normalize_embeddings", False)
    return model


@lru_cache(maxsize=5)
def get_embedding_model_cached(method: str,
                               yaml_path: str = DEFAULT_YAML_PATH):
    return get_model(method, yaml_path)
