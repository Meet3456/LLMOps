from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_question_prompt = ChatPromptTemplate.from_messages([
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
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}

