import base64
import os
from groq import AsyncGroq
from multi_doc_chat.logger import GLOBAL_LOGGER as log
import asyncio
import concurrent.futures
from typing import Dict, Any

client = None

def get_client():
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY_COMPOUND")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        client = AsyncGroq(api_key=api_key)
    return client

# Global semaphore to limit concurrent Groq calls
semaphore = asyncio.Semaphore(5)

# shared thread pool exector
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


async def _read_file_b64(path: str) -> str:
    """Asynchronously read a file and encode it as base64."""

    def _read():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    return await asyncio.get_running_loop().run_in_executor(executor, _read)


async def encode_image_b64(path: str) -> str:
    """Alias that uses shared executor."""
    return await _read_file_b64(path)


async def _caption_request(
    b64: str,
    prompt: str,
    timeout: int,
    model: str,
    max_tokens: int,
    top_p: float,
    temperature: float,
):
    client = get_client()

    message = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
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

        return completion


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
            log.debug(f"Captioning image from bytes, attempt {attempt}/{retries}")

            response = await asyncio.wait_for(
                _caption_request(
                    base_64, prompt, timeout, model, max_tokens, top_p, temperature
                ),
                timeout=timeout + 5,
            )

            message = response.choices[0].message

            caption_text = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            log.debug(f"Captioning successful: {caption_text}")
            return {"caption": caption_text, "raw_message": message}
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
    """Async captioning from an image file path using executor for file read."""
    try:
        b64 = await _read_file_b64(image_path)
        return await caption_image_from_bytes(
            base64.b64decode(b64), prompt=prompt, **kwargs
        )
    except Exception as e:
        log.error("Image captioning failed", error=str(e), path=image_path)
        return {"caption": "", "error": str(e)}
