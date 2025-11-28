from colorama import init

from CustomLogger import log, LoggerCategory, CustomLogger
from CustomRag import CustomRag, DEFAULT_MODEL
from TestRunner import TestRunner
from utils import load_test_set

if __name__ == "__main__":
    CustomLogger.configure([LoggerCategory.ERROR, LoggerCategory.PROCESSING_QUESTION])
    init(autoreset=True)
    rag = CustomRag()
    # rag.clear_vectorstore()
    # rag.load_pdf_files(use_semantic=True)
    test_runner = TestRunner(rag, DEFAULT_MODEL)
    file_name = "questions_rfc6265"
    questions = load_test_set(f"tests/questions/{file_name}.json")

    for index, details in enumerate(questions):
        question = details.get("question", "")
        expected_answer = details.get("expected_answer", "")
        keywords = details.get("keywords", [])
        log.processing_question(
            f"Question {index + 1}: {question}, Expected Answer: {expected_answer}, Keywords: {keywords}")
        test_runner.run_test(question, expected_answer, keywords)

    test_runner.save_tests_results(file_name)
