from .custom_logger import get_logger

GLOBAL_LOGGER = get_logger("multi_doc_chat")

__all__ = ["GLOBAL_LOGGER"]
