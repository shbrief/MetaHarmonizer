from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import yaml
from functools import lru_cache
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from src.models.llm_reranker import LLMReranker

load_dotenv()

DEFAULT_YAML_PATH = os.getenv("METHOD_MODEL_YAML",
                              "src/models/method_model.yaml")
CACHE_ROOT = "model_cache"


def load_method_model_dict(yaml_path: str = DEFAULT_YAML_PATH) -> dict:
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)


def get_model(
        method: str,
        yaml_path: str = DEFAULT_YAML_PATH,
        model_type: str = "embedding") -> SentenceTransformer | CrossEncoder:
    """
    Load a model from Hugging Face Hub (cached locally).
    
    Parameters:
    - method: Method name as defined in YAML
    - yaml_path: Path to method-model mapping YAML
    - model_type: Either "embedding" or "reranker"
    
    Returns:
    - SentenceTransformer for embedding models
    - CrossEncoder for reranker models
    """
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load different model types
    if model_type == "embedding":
        model = SentenceTransformer(local_dir,
                                    device=device,
                                    local_files_only=True)
        model._is_normalized = getattr(model, "normalize_embeddings", False)
    elif model_type == "reranker":
        model = CrossEncoder(local_dir, max_length=512, device=device)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Must be 'embedding' or 'reranker'"
        )

    return model


@lru_cache(maxsize=5)
def get_embedding_model_cached(method: str,
                               yaml_path: str = DEFAULT_YAML_PATH):
    return get_model(method, yaml_path, model_type="embedding")


@lru_cache(maxsize=5)
def get_reranker_model_cached(method: str, yaml_path: str = DEFAULT_YAML_PATH):
    return get_model(method, yaml_path, model_type="reranker")


@lru_cache(maxsize=2)
def get_llm_reranker_cached(method: str,
                            use_8bit: bool = False,
                            batch_size: int = 4,
                            yaml_path: str = DEFAULT_YAML_PATH):
    """
    Get cached LLM reranker model.
    
    Args:
        method: Model identifier (defined in YAML)
        use_8bit: Use 8-bit quantization
        batch_size: Inference batch size
        yaml_path: Path to method-model mapping YAML
    """
    method_model_dict = load_method_model_dict(yaml_path)

    if method not in method_model_dict:
        raise ValueError(
            f"Unknown LLM reranker method: {method}. Must be one of {list(method_model_dict.keys())}"
        )

    model_name = method_model_dict[method]

    return LLMReranker(model_name=model_name,
                       use_8bit=use_8bit,
                       batch_size=batch_size,
                       cache_dir=CACHE_ROOT)
