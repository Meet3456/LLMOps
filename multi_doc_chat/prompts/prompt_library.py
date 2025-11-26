from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Router Prompt(LLM Based):LLM will decide where to route the query(rag/reasoning/tools)
router_system_prompt = (
    "You are an expert router. Choose the SINGLE best handler for the user query.\n\n"
    "You MUST answer with a JSON object: {\"datasource\": \"rag\" | \"tools\" | \"reasoning\"}.\n\n"
    "ROUTES:\n"
    "- \"rag\": Use when the user is likely asking about uploaded documents, PDFs, statements, invoices,\n"
    "          or anything that can be answered from internal indexed content.\n"
    "- \"tools\": Use when the query clearly needs external live data (news, weather, prices, stock),\n"
    "            web browsing, code execution, or non-trivial math solving.\n"
    "- \"reasoning\": Use for deep explanations, conceptual questions, teaching, reasoning, or anything\n"
    "               that is not clearly document-based or tool-based.\n\n"
    "You are given some routing signals. They are only hints, not hard rules.\n"
    "Be strict about returning VALID JSON only." 
)

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",router_system_prompt),
        (
            "human",
            "User Query: \n{input}\n"
            "Routing signals(JSON): \n{signals}\n"
            "Strictly Return the best route as JSON , Do not add extra text."
        )
    ]

)


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

