from .custom_logger import CustomLogger

# Global logger instance for entire project
GLOBAL_LOGGER = CustomLogger().get_logger()

__all__ = ["GLOBAL_LOGGER"]
