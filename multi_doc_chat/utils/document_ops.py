from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from fastapi import UploadFile

from multi_doc_chat.utils.vision import caption_image, caption_image_from_bytes
from multi_doc_chat.utils.table import (
    extract_tables_from_pdf,
    extract_tables_from_csv,
    html_tables_to_json,
)

import fitz
from bs4 import BeautifulSoup
import concurrent.futures
import json

executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


async def _process_single_path(
    p: Path, images_dir: Path, tables_dir: Path
) -> List[Document]:
    """
    Loads text docs and also extracts images + tables into Documents.
    Each Document.metadata will include 'modality' in {'text','table','image'} and 'source'.
    """
    docs: List[Document] = []
    extension = p.suffix.lower()
    try:
        if extension == ".pdf":
            loader = PyPDFLoader(str(p))
            text_docs = await asyncio.get_running_loop().run_in_executor(
                executor, loader.load
            )

            # mark the metadata
            for doc in text_docs:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata.update({"modality": "text", "source": str(p)})
            docs.extend(text_docs)

            # extract tables via camelot in executor
            tables = await asyncio.get_running_loop().run_in_executor(
                executor, extract_tables_from_pdf, str(p)
            )

            for idx, t in enumerate(tables):
                # Save CSV
                csv_path = tables_dir / f"{p.stem}_page{t['page']}_table{idx}.csv"
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(t["csv"])

                # Save JSON
                json_path = tables_dir / f"{p.stem}_page{t['page']}_table{idx}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(t["json"], f, indent=2)

                table_text = (
                    f"Extracted table from page {t['page']}.\n"
                    f"Preview:\n" + "\n".join(t["csv"].splitlines()[:10])
                )

                docs.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "modality": "table",
                            "source": str(p),
                            "saved_table_csv": str(csv_path),
                            "saved_table_json": str(json_path),
                            "page": t["page"],
                        },
                    )
                )

            # extract images using pymupdf and caption them concurrently from bytes
            pdf = fitz.open(str(p))
            caption_tasks = []
            image_meta = []
            for i in range(len(pdf)):
                page = pdf[i]
                image_list = page.get_images(full=True)
                for img_index, img_meta in enumerate(image_list):
                    try:
                        xref = img_meta[0]
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Save image to disk
                        img_path = (
                            images_dir / f"{p.stem}_page{i + 1}_img{img_index + 1}.png"
                        )
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)

                        caption_tasks.append(caption_image_from_bytes(image_bytes))
                        image_meta.append(
                            {
                                "source": str(p),
                                "page": i + 1,
                                "image_index": img_index + 1,
                                "saved_image_path": str(img_path),
                            }
                        )
                    except Exception as e:
                        log.warning(
                            "extract_image failed",
                            file=str(p),
                            page=i + 1,
                            error=str(e),
                        )

            if caption_tasks:
                captions = await asyncio.gather(*caption_tasks)
                for cap_result, meta in zip(captions, image_meta):
                    if cap_result.get("caption"):
                        docs.append(
                            Document(
                                page_content=cap_result["caption"],
                                metadata={
                                    "modality": "image",
                                    "source": meta["source"],
                                    "page": meta["page"],
                                },
                            )
                        )
            pdf.close()

        elif extension == ".docx":
            loader = Docx2txtLoader(str(p))
            text_docs = await asyncio.get_running_loop().run_in_executor(
                executor, loader.load
            )
            for d in text_docs:
                d.metadata = dict(d.metadata or {})
                d.metadata.update({"modality": "text", "source": str(p)})
            docs.extend(text_docs)

        elif extension == ".txt":
            loader = TextLoader(str(p), encoding="utf-8")
            text_docs = await asyncio.get_running_loop().run_in_executor(
                executor, loader.load
            )
            for d in text_docs:
                d.metadata = dict(d.metadata or {})
                d.metadata.update({"modality": "text", "source": str(p)})
            docs.extend(text_docs)

        elif extension == ".csv":
            tables = await asyncio.get_running_loop().run_in_executor(
                executor, extract_tables_from_csv, str(p)
            )
            for t in tables:
                docs.append(
                    Document(
                        page_content=t["csv"],
                        metadata={
                            "modality": "table",
                            "source": str(p),
                            "table_json": t["json"],
                        },
                    )
                )

        elif extension in {".html", ".htm"}:
            tables = await asyncio.get_running_loop().run_in_executor(
                executor, html_tables_to_json, str(p)
            )
            for t in tables:
                docs.append(
                    Document(
                        page_content=t["csv"],
                        metadata={
                            "modality": "table",
                            "source": str(p),
                            "table_json": t["json"],
                        },
                    )
                )

            def _read_html_text():
                with open(p, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    return soup.get_text(separator="\n")

            body_text = await asyncio.get_running_loop().run_in_executor(
                executor, _read_html_text
            )
            docs.append(
                Document(
                    page_content=body_text,
                    metadata={"modality": "text", "source": str(p)},
                )
            )

        elif extension in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            # single image file, caption it (we can pass path to caption_image which reads file in executor)
            cap = await caption_image(str(p))
            if cap.get("caption"):
                docs.append(
                    Document(
                        page_content=cap["caption"],
                        metadata={
                            "modality": "image",
                            "source": str(p),
                            "image_path": str(p),
                        },
                    )
                )

        else:
            log.warning("Unsupported extension skipped", path=str(p))

    except Exception as e:
        log.error("Failed processing file", file=str(p), error=str(e))
        raise DocumentPortalException(
            f"Failed processing file {str(p)}: {str(e)}"
        ) from e

    return docs


async def load_documents_and_assets(
    paths: Iterable[Path], images_dir: Path, tables_dir: Path
) -> List[Document]:
    """Process all paths concurrently and return flattened Document list."""
    try:
        tasks = [_process_single_path(p, images_dir, tables_dir) for p in paths]
        results = await asyncio.gather(*tasks)

        # flatten the list of lists
        all_docs = [doc for doc_list in results for doc in doc_list]
        log.info("Documents & assets loaded", count=len(all_docs))

        for doc in all_docs:
            if doc.metadata.get("modality") == "image":
                print("\n Extracted Image Caption:")
                print("Source:", doc.metadata.get("source"))
                print("Page:", doc.metadata.get("page"))
                # print("Caption:", doc.page_content)

        return all_docs

    except Exception as e:
        log.error("Failed loading documents/assets", error=str(e))
        raise DocumentPortalException("Error loading documents/assets", e) from e
