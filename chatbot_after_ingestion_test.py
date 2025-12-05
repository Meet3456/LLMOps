import traceback

from multi_doc_chat.graph.orchestrator import Orchestrator
from multi_doc_chat.logger import GLOBAL_LOGGER as log


def print_block(title, content):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(content)
    print("=" * 80 + "\n")


def test_formatting():
    pass


def testing_again():
    pass


def run_end_to_end(index_path: str):
    try:
        print_block("STARTING END-TO-END MULTI-TURN TEST", "")

        orch = Orchestrator(index_path=index_path)
        print("Orchestrator loaded successfully.\n")

        # Chat history stores langchain-style messages
        chat_history = []

        # Define test turns
        conversation_turns = [
            # " All Lamels are Signots with buttons.No yellow Signots have buttons.No Lamels are yellow. If the first two statements are true, the third statement is - true , false , uncertain",
            # "Look at this series: 21, 9, 21, 11, 21, 13, 21, ... What number should come next?"
            # "what hurts the transformer model quality",
            # "Look at this series: 21, 9, 21, 11, 21, 13, 21, ... What number should come next?",
            # "what hurts the model quality with respect to model variations in transformers",
            # "Summarize the table , Table 2: The Transformer achieves better BLEU scores",
            # "Details about the Optimizer",
            # "BLEU scores of English-to-French newstest2014 test",
            # "summarise Results are on Section 23 of WSJ"
            # "list all the parsers,there training and resulting score"
            # "who recently scored a century against SA in odi match"
            # "give details about Multi-Head Attention image"
            # "what was the dropout rate of Transformer model trained for English-to-French",
            "summarize the Machine Translation section"
            # "list all the parsers,there training and resulting score",
            # "who recently scored a century against SA in odi match",
            # "detail summary of machine translation section",
            # "scorecard or recent ind vs sa odi match"
            "no of players who scored century in recent ind vs sa match"
        ] 

        for i, user_query in enumerate(conversation_turns, start=1):
            print_block(f"🧠 TURN {i}: USER QUERY", user_query)

            # 1. ROUTE THE QUERY
            route = orch.route_query(user_query, chat_history)
            print(f"→ Router decision: {route}")

            # 2. EXECUTE PIPELINE
            if route == "rag":
                output = orch.run_rag(user_query, chat_history)
            elif route == "reasoning":
                output = orch.run_reasoning(user_query)
            elif route == "tools":
                output = orch.run_tools(user_query)
            else:
                output = f"[ERROR] Invalid route: {route}"

            # 3. Print output
            print_block("SYSTEM RESPONSE", output)

            # 4. Save to history (LangChain message simulation)
            chat_history.append({"role": "human", "content": user_query})
            chat_history.append({"role": "assistant", "content": output})

        print_block("🎉 END-TO-END TEST COMPLETED SUCCESSFULLY", "")

    except Exception as e:
        print_block("❌ ERROR DURING END-TO-END TEST", str(e))
        print(traceback.format_exc())


if __name__ == "__main__":
    # Change index path according to your setup
    DEFAULT_INDEX = "faiss_index/session_04_dec_2025_8:04_pm_8a88"

    run_end_to_end(DEFAULT_INDEX)
