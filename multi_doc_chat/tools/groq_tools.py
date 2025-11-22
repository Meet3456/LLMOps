from groq import Groq
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
import sys

class GroqToolClient:
    """
    Multi-key failover client for Compound / Compound-Mini.
    Handles:
    - key rotation
    - reasoning_format safety
    - avoids invalid Groq parameter combos
    """

    def __init__(self, api_keys: list[str]):
        valid = [k for k in api_keys if k]
        if not valid:
            raise DocumentPortalException("No valid Groq API keys provided", sys)

        self.api_keys = valid
        self.idx = 0
        self.client = self._make(valid[self.idx])

        log.info("GroqToolClient initialized", keys=len(valid))

    def _make(self, key: str) -> Groq:
        return Groq(api_key=key)

    def _rotate(self):
        self.idx = (self.idx + 1) % len(self.api_keys)
        self.client = self._make(self.api_keys[self.idx])
        log.warning("Rotated Groq API key", new_index=self.idx)

    def call_compound(self, user_prompt, model, enabled_tools, max_tokens, stream=False):
        """
        Only valid compound params:
        - reasoning_format (parsed | raw | hidden)
        - NO include_reasoning
        """
        for attempt in range(len(self.api_keys)):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    compound_custom={
                        "tools": {
                            "enabled_tools": enabled_tools,
                            "wolfram_settings": {"authorization": "HP58J63QR8"},
                        }
                    },
                    reasoning_format="parsed",
                    max_tokens=max_tokens,
                    stream=stream,
                )

                msg = resp.choices[0].message
                return {
                    "content": msg.content,
                    "reasoning": getattr(msg, "reasoning", None),
                    "executed_tools": getattr(msg, "executed_tools", None),
                }

            except Exception as e:
                log.warning("Compound call failed", error=str(e))
                self._rotate()

        raise DocumentPortalException("All Groq API keys failed during compound call")
