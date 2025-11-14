import base64
import os
from groq import Groq
from multi_doc_chat.logger import GLOBAL_LOGGER as log

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def caption_image(image_path: str):
    try:
        pass
    except Exception as e:
        pass