from pathlib import Path
import os
import yaml

# will return the root directory of the project => multi-doc-chat

def _project_root() -> Path:
    # __file__ = Holds the path of the current file
    # Path(__file__) = Converts the file path to a Path object representing current file's path
    # resolve() = Converts that relative path into an absolute path and resolves any symbolic links (.., . etc.).
    # The [.parents] property gives you a list-like sequence of all parent directories:
    return Path(__file__).resolve().parents[1]

def load_config(config_path: str | None = None) -> dict:
    
    env_path = os.getenv("CONFIG_PATH", None)

    if config_path is None:
        config_path = env_path or str(_project_root() / "multi_doc_chat" / "config" / "config.yaml")

    path = Path(config_path)

    if not path.is_absolute():
        path = _project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path , "r" , encoding="utf-8") as file:
        return yaml.safe_load(file) or {}