"""
Tool detection logic to determine if a query needs external tools.
"""
import re
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class ToolDetector:
    """
    Determines if a query requires external tool execution.
    Uses keyword matching and pattern detection.
    """
    
    TOOL_KEYWORDS = {
        "web_search": ["search", "find online", "look up", "latest", "external" , "information" , "current", "news", "today"],
        "code_interpreter": ["calculate", "compute", "run code", "execute", "python", "code"],
        "wolfram_alpha": ["solve", "equation", "math", "calculate", "integral", "derivative"],
        "browser": ["visit", "open website", "go to", "browse"],
    }
    
    def __init__(self):
        log.info("ToolDetector initialized")
    
    def needs_tools(self, query: str) -> bool:
        """
        Check if query requires external tools.
        
        Args:
            query: User's query string
            
        Returns:
            True if tools are needed, False otherwise
        """
        query_lower = query.lower()
        
        # Check for tool-related keywords
        for tool_type, keywords in self.TOOL_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                log.info("Tool needed", tool_type=tool_type, query=query[:50])
                return True
        
        # Check for URLs (might need browser automation)
        if re.search(r'https?://', query):
            log.info("URL detected, tool needed", query=query[:50])
            return True
        
        # Check for mathematical expressions
        if re.search(r'[\d\+\-\*/\^\(\)=]', query) and any(word in query_lower for word in ["solve", "calculate", "compute"]):
            log.info("Math expression detected, tool needed")
            return True
        
        log.info("No tools needed", query=query[:50])
        return False