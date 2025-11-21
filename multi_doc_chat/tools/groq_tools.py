from groq import Groq
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException

class GroqToolClient:

    def __init__(self , api_key: str):
        try:
            self.client = Groq(
                api_key=api_key,
                default_headers={"Grow-Model-Version":"latest"}
            )
            log.info("Groq client initialized successfully")
        except Exception as e:
            log.error("Failed to initialize Groq client", error=str(e))
            raise DocumentPortalException("Groq client initialization error", e)
        
    def call_compound(
        self,
        user_prompt: str,
        model: str,
        enabled_tools: list,
        max_tokens: int,
        stream=False,
    ):
        '''
            Args:
                user_prompt: The user's query
                model: Model name (e.g., "groq/compound-mini")
                enabled_tools: List of tools to enable
                reasoning_format: How to format reasoning output
                include_reasoning: Whether to include reasoning traces
                stream: Whether to stream the response
                
            Returns:
                Dict with content, reasoning, and executed_tools
        '''
        
        log.info("Groq tool call", model_used=model, tools=enabled_tools, prompt=user_prompt)

        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            compound_custom={
                "tools": {
                    "enabled_tools": enabled_tools,
                    "wolfram_settings": {"authorization": "HP58J63QR8"},
                }
            },
            max_tokens=max_tokens,
            stream=stream,
        )
        msg = resp.choices[0].message
        
        return {
            "content": msg.content,
            "reasoning": getattr(msg, "reasoning", None),
            "executed_tools": getattr(msg, "executed_tools", None),
        }