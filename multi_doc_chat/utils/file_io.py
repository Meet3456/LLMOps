from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import List, Union,Iterable
from multi_doc_chat.logger.custom_logger import CustomLogger
from multi_doc_chat.exception.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3",".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# Local logger instance
log = CustomLogger().get_logger(__name__)

def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            # getting the file name
            name = getattr(uf, "filename", getattr(uf, "name", "file"))
            # getting the file extension
            extension = Path(name).suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                log.warning(f"Unsupported file type: {extension} for file {name}. Skipping.")
                continue

            # Clean file name (only alphanum, dash, underscore)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(name).stem).lower()
            file_name = f"{safe_name}_{uuid.uuid4().hex[:5]}{extension}"
            output_path = target_dir / file_name

            with open(output_path, "wb") as f:
                # Prefer underlying file buffer when available (e.g., Starlette UploadFile.file)
                if hasattr(uf, "file") and hasattr(uf.file, "read"):
                    f.write(uf.file.read())
                elif hasattr(uf, "read"):
                    data = uf.read()
                    # If a memoryview is returned, convert to bytes; otherwise assume bytes
                    if isinstance(data, memoryview):
                        data = data.tobytes()
                    f.write(data)
                else:
                    # Fallback for objects exposing getbuffer():
                    buffer = getattr(uf, "getbuffer", None)
                    if callable(buffer):
                        data = buffer()
                        if isinstance(data, memoryview):
                            data = data.tobytes()
                        f.write(data)
                    else:
                        raise ValueError("Unsupported uploaded file object; no readable interface")

            saved.append(output_path)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(output_path))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e
    