import time
from colorama import Fore, init

from CustomRag import CustomRag, s_i
from utils.log_utils import s_q, s_a

if __name__ == "__main__":
    init(autoreset=True)
    rag = CustomRag()

    # rag.clear_vectorstore()
    # rag.load_text_files()
    # rag.load_pdf_files(use_semantic=True)

    print(f"{s_i} Hello! How can I assist you today?")
    while True:
        user_input = input(f"{s_q} Question: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        start_time = time.time()
        answer, _, _ = rag.ask(user_input)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{Fore.YELLOW + s_i} Execution time: {elapsed_time:.2f} seconds")
        print(f"{s_a} RAG:\n{Fore.MAGENTA + answer}")
