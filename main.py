from CustomRag import CustomRag, s_i
from utils.log_utils import s_q, s_a

if __name__ == "__main__":
    rag = CustomRag()

    # rag.load_universe_files()
    rag.load_rfc_files()

    print(f"{s_i} Hello! How can I assist you today?")
    while True:
        user_input = input(f"{s_q} Question: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = rag.ask(user_input)
        print(f"{s_a} RAG: {answer}")
