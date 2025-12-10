from colorama import init

from utils.CustomLogger import LoggerCategory, CustomLogger
from core.CustomRag import CustomRag, DEFAULT_MODEL
from core.TestRunner import TestRunner
from utils.utils import load_test_set

if __name__ == "__main__":
    # CustomLogger.configure([LoggerCategory.ERROR, LoggerCategory.PROCESSING_QUESTION, LoggerCategory.FULL_PROMPT, LoggerCategory.FULL_RESPONSE])
    CustomLogger.configure([LoggerCategory.ERROR, LoggerCategory.PROCESSING_QUESTION])
    init(autoreset=True)
    rag = CustomRag()
    # rag.clear_vectorstore()
    # rag.load_pdf_files(use_semantic=True)
    test_runner = TestRunner(rag, DEFAULT_MODEL)
    file_name = "questions_rfc6265"
    questions = load_test_set(f"tests/questions/{file_name}.json")

    for index, details in enumerate(questions):
        test_runner.run_test(details)

    test_runner.save_tests_results(file_name)

    # test_runner.multirun_tests(questions, 10)
