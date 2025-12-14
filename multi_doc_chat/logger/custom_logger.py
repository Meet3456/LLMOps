import logging
from rich.console import Console
from rich.logging import RichHandler

# -------------------------------------------------
# Console with proper color handling
# -------------------------------------------------
console = Console(force_terminal=True, color_system="truecolor")

# -------------------------------------------------
# Root logging config
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
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
)

# -------------------------------------------------
# Silence noisy libraries
# -------------------------------------------------
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)

logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# -------------------------------------------------
# Export logger
# -------------------------------------------------
def get_logger(name: str):
    return logging.getLogger(name)
