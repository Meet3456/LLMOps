import asyncio
import base64
import concurrent.futures
import os
from typing import Any, Dict, List

from groq import AsyncGroq

from multi_doc_chat.logger import GLOBAL_LOGGER as log

client = None


def get_client(): 
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY_COMPOUND")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        client = AsyncGroq(api_key=api_key)
    return client


MAX_GROQ_CONCURRENCY = int(os.getenv("MAX_GROQ_CONCURRENCY", 12))

# Global semaphore to limit concurrent Groq calls
semaphore = asyncio.Semaphore(MAX_GROQ_CONCURRENCY)

# shared thread pool exector
executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)


async def _read_file_b64(path: str) -> str:
    """Asynchronously read a file and encode it as base64."""

    def _read():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _read)


async def encode_image_b64(path: str) -> str:
    """Alias that uses shared executor."""
    return await _read_file_b64(path)


async def _caption_request(
    b64_List: List[str],
    prompt: str,
    model: str,
    max_tokens: int,
    top_p: float,
    temperature: float,
) -> List[str]:
    client = get_client()

    async def _process_single_image(b64: str) -> str:
        message = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
        ]

        async with semaphore:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                response_format={"type": "text"},
            )

        msg = completion.choices[0].message

        if isinstance(msg.content, str):
            return msg.content
        return str(msg.content)

    tasks = [_process_single_image(single_b64) for single_b64 in b64_List]

    return await asyncio.gather(*tasks, return_exceptions=True)


async def caption_image_from_bytes(
    image_bytes: bytes,
    prompt: str = "Describe the image in a concise caption. Include objects, scene, and any notable attributes.",
    retries: int = 2,
    timeout: int = 30,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens: int = 512,
    top_p: float = 0.9,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    base_64 = base64.b64encode(image_bytes).decode("utf-8")

    for attempt in range(1, retries + 1):
        try:
            log.debug(
                "Captioning image from bytes",
                attempt=attempt,
                retries=retries,
            )

            captions = await _caption_request(
                [base_64],
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
            )

            caption_text = captions[0] if captions else ""
            log.debug("Captioning successful", caption=caption_text)
            return {"caption": caption_text}
        except asyncio.TimeoutError:
            log.warning("Groq caption timeout", attempt=attempt)
        except Exception as e:
            log.warning("Groq caption error", attempt=attempt, error=str(e))

    return {"caption": "", "error": "caption_failed"}


async def caption_image(
    image_path: str,
    prompt: str = "Describe the image in a concise caption. Include objects, scene, and any notable attributes.",
    **kwargs,
) -> Dict[str, Any]:
    """
    Async captioning from an image file path.
    Reads the file bytes in a thread pool, then calls caption_image_from_bytes.
    """
    try:
        loop = asyncio.get_running_loop()

        def _read_bytes():
            with open(image_path, "rb") as f:
                return f.read()

        image_bytes = await loop.run_in_executor(executor, _read_bytes)

        return await caption_image_from_bytes(image_bytes, prompt=prompt, **kwargs)

    except Exception as e:
        log.error(
            "Image captioning failed",
            error=str(e),
            path=image_path,
        )
        return {"caption": "", "error": str(e)}
