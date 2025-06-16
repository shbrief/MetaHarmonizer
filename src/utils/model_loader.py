from sentence_transformers import SentenceTransformer
import torch
import yaml
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_YAML_PATH = os.getenv("METHOD_MODEL_YAML",
                              "src/models/method_model.yaml")


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
    model_name = method_model_dict[method]
    return SentenceTransformer(
        model_name, device='cuda' if torch.cuda.is_available() else 'cpu')


@lru_cache(maxsize=5)
def get_embedding_model_cached(method: str,
                               yaml_path: str = DEFAULT_YAML_PATH):
    return get_model(method, yaml_path)
