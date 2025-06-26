import os
from huggingface_hub import snapshot_download
from src.utils.model_loader import DEFAULT_YAML_PATH, load_method_model_dict


def cache_models(methods, cache_root):
    model_map = load_method_model_dict(DEFAULT_YAML_PATH)
    for method in methods:
        repo_id = model_map.get(method)
        if repo_id is None:
            print(f"Chosen method '{method}' is not found in the model map.")
            continue

        target_dir = os.path.join(cache_root, method)
        os.makedirs(target_dir, exist_ok=True)

        print(
            f"Starting to cache model for method: {method} (repo_id: {repo_id})"
        )
        local_dir = snapshot_download(repo_id=repo_id,
                                      cache_dir=cache_root,
                                      local_dir=target_dir,
                                      repo_type="model",
                                      local_files_only=False)
        print(f"Model for method '{method}' cached at: {local_dir}")


if __name__ == "__main__":
    methods_to_cache = ["pubmed-bert", "sap-bert", "mt-sap-bert"]

    cache_root = os.getenv("MODEL_CACHE_DIR", "./model_cache")
    os.makedirs(cache_root, exist_ok=True)

    cache_models(methods_to_cache, cache_root)
