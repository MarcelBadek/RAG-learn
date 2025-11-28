import time
from colorama import init

from CustomLogger import log
from CustomRag import CustomRag

if __name__ == "__main__":
    init(autoreset=True)
    rag = CustomRag()

    # rag.clear_vectorstore()
    # rag.load_text_files()
    # rag.load_pdf_files(use_semantic=True)

    log.answer(f"Hello! How can I assist you today?")
    while True:
        user_input = input(f"Question: ")
        if user_input.lower() in {"exit", "quit"}:
            log.answer("Goodbye!")
            break

        start_time = time.time()
        answer, _, _ = rag.ask(user_input)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.statistics(f"Execution time: {elapsed_time:.2f} seconds")
        log.answer(answer)
