import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

app_env = os.getenv("APP_ENV", "production").lower()

# Console with proper color handling
console = Console(force_terminal=True, color_system="truecolor")


# Configure logging strategy as per the production environment
def configure_logging():
    log_level = logging.INFO

    if app_env == "production":
        # Force re-configuration using force=True (Python 3.8+)
        logging.basicConfig(
            level=log_level,
            # Standard format: Time | Level | Logger Name | Message
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )

    else:
        # Root logging config
        logging.basicConfig(
            level=log_level,
            format="%(message)s",  # Rich handles formatting
            datefmt="[%H:%M:%S.%f]",
            handlers=[
                RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    tracebacks_show_locals=False,
                    show_time=True,
                    show_level=True,
                    show_path=True,
                    log_time_format="%H:%M:%S.%f",
                )
            ],
            force=True,
        )

    # Silence noisy libraries
    silenced_loggers = [
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "sqlalchemy.dialects",
        "sqlalchemy.orm",
        "uvicorn.access",
        "httpx",
        "urllib3",
        "botocore",
        "boto3",
    ]

    for logger_name in silenced_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Specific adjustments for Uvicorn
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


# Initialize configuration immediately when module is imported
configure_logging()


# Export logger
def get_logger(name: str):
    return logging.getLogger(name)
