from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Router Prompt(LLM Based):LLM will decide where to route the query(rag/reasoning/tools)
router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert router. Decide which subsystem should handle the query.\n"
            "OPTIONS:\n"
            "- 'rag' → If the query relates to internal documents, research papers, PDFs, or indexed data.\n"
            "- 'tools' → If the query needs external information, latest news, URLs, math, or computation.\n"
            "- 'reasoning' → If the query requires thinking, explanation, logic, analysis, or general questions.\n\n"
            "You will receive:\n"
            "- input: The user query\n"
            "- signals: Routing signals as JSON\n\n"
            "ONLY respond using valid JSON following this schema:\n" 

            
            "{{ \"source\": \"rag\" | \"tools\" | \"reasoning\" }}\n"
        )
    ),
    ("human", "Query: {input}\n Signals: {signals}")
])



# Prompt for rewriting questions(queries) with context
contextualize_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a query rewriting assistant.\n"
            "Your task is to rewrite the user’s latest question into a fully standalone question.\n"
            "Use the chat history only to fill missing references (e.g., pronouns, 'this', 'that', etc.).\n"
            "Do NOT answer the question.\n"
            "Do NOT add information.\n"
            "DO NOT change meaning.\n"
            "Return ONLY the rewritten question as plain text.\n"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ]
)


# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a helpful assistant that must answer using ONLY the provided context.\n"
            "If the answer is not in the context, say: \"I don't know based on the available documents.\"\n"
            "Be concise and professional.\n\n"
            "Context:\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "router": router_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}

