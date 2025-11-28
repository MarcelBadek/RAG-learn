import datetime
import os

from colorama import Fore

from AdjustedOllama import AdjustedOllama

from CustomLogger import log
from utils import save_json


class TestRunner:
    def __init__(self, rag_instance, model):
        self.rag = rag_instance
        self.adjusted_model = AdjustedOllama(model)
        self.tests_results = {}

        if not os.path.exists("tests/results"):
            os.makedirs("tests/results")

    def run_test(self, question, expected_answer, keywords):
        current_test_number = len(self.tests_results) + 1
        answer, docs, details = self.rag.ask(question)

        correct_context = self.adjusted_model.validate_answer_with_context(answer, "\n\n".join(docs))

        correct_expected_answer = False
        correct_keywords = False

        if expected_answer != "":
            correct_expected_answer = self.adjusted_model.validate_answer_with_expected_answer(answer, expected_answer)
        else:
            log.error(f"Question: \"{question}\" is missing an expected answer.")

        if keywords and len(keywords) > 0:
            correct_keywords = self.adjusted_model.validate_answer_with_expected_keywords(answer, keywords)
        else:
            log.error(f"Question: \"{question}\" is missing expected keywords.")

        self.tests_results[current_test_number] = {
            "question": question,
            "details": details,
            "expected_answer": expected_answer,
            "keywords": keywords,
            "answer": answer,
            "is_correct_based_on_context": correct_context,
            "is_correct_based_on_expected_answer": correct_expected_answer,
            "is_correct_based_on_keywords": correct_keywords,
            "used_documents": docs
        }

        return answer

    def multirun_tests(self, test_set, run_number):
        return


    def save_tests_results(self, base_filename="test_results"):
        current_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_details = f"{base_filename}_{current_date_time}_details.json"
        filename_simple = f"{base_filename}_{current_date_time}.json"
        filename_statistics = f"{base_filename}_{current_date_time}_summary.json"

        save_json(self.tests_results, f"tests/results/{filename_details}")
        simple_results = {
            test_num: {
                "question": result["question"],
                "answer": result["answer"],
            }
            for test_num, result in self.tests_results.items()
        }
        save_json(simple_results, f"tests/results/{filename_simple}")

        self.show_and_save_summary(filename_statistics)

    def show_and_save_summary(self, filename):
        fully_correct = []
        mostly_correct = []
        partially_correct = []
        incorrect = []
        response_time = 0
        token_usage = 0
        total_tests = len(self.tests_results)
        correct_context = sum(1 for result in self.tests_results.values() if result["is_correct_based_on_context"])
        correct_expected_answer = sum(
            1 for result in self.tests_results.values() if result["is_correct_based_on_expected_answer"])
        correct_keywords = sum(1 for result in self.tests_results.values() if result["is_correct_based_on_keywords"])

        for result in self.tests_results.values():
            correct_count = sum([
                result["is_correct_based_on_context"],
                result["is_correct_based_on_expected_answer"],
                result["is_correct_based_on_keywords"]
            ])
            if correct_count == 3:
                fully_correct.append(result["question"])
            elif correct_count == 2:
                mostly_correct.append(result["question"])
            elif correct_count == 1:
                partially_correct.append(result["question"])
            else:
                incorrect.append(result["question"])

            response_time = response_time + float(result["details"].get("total_duration_s", 0))
            token_usage = token_usage + int(result["details"].get("prompt_eval_count", 0))

        average_response_time = response_time / total_tests if total_tests > 0 else 0
        average_token_usage = token_usage / total_tests if total_tests > 0 else 0

        log.always(f"Total Tests: {total_tests}")
        log.always(f"Average Response Time: {Fore.LIGHTBLUE_EX}{average_response_time:.2f} seconds")
        log.always(f"Total Prompt Tokens Used: {Fore.LIGHTBLUE_EX}{token_usage}")
        log.always(f"Average Prompt Tokens per Test: {Fore.LIGHTBLUE_EX}{average_token_usage:.2f}")
        log.always(
            f"Correct based on Context: {Fore.LIGHTBLUE_EX}{correct_context} ({self._calculate_percentage_in_total_tests(correct_context):.2f}%)")
        log.always(
            f"Correct based on Expected Answer: {Fore.LIGHTBLUE_EX}{correct_expected_answer} ({self._calculate_percentage_in_total_tests(correct_expected_answer):.2f}%)")
        log.always(
            f"Correct based on Keywords: {Fore.LIGHTBLUE_EX}{correct_keywords} ({self._calculate_percentage_in_total_tests(correct_keywords):.2f}%)")
        log.always(
            f"Fully Correct (correct 3/3): {Fore.LIGHTBLUE_EX}{len(fully_correct)} ({self._calculate_percentage_in_total_tests(len(fully_correct)):.2f}%)")
        log.always(
            f"Mostly Correct (correct 2/3): {Fore.LIGHTBLUE_EX}{len(mostly_correct)} ({self._calculate_percentage_in_total_tests(len(mostly_correct)):.2f}%)")
        log.always(
            f"Partially Correct (correct 1/3): {Fore.LIGHTBLUE_EX}{len(partially_correct)} ({self._calculate_percentage_in_total_tests(len(partially_correct)):.2f}%)")
        log.always(
            f"Incorrect (correct 0/3): {Fore.LIGHTBLUE_EX}{len(incorrect)} ({self._calculate_percentage_in_total_tests(len(incorrect)):.2f}%)")

        statistics = {
            "total_tests": total_tests,
            "response_time_average": average_response_time,
            "total_prompt_tokens_used": token_usage,
            "average_prompt_tokens_per_test": average_token_usage,
            "correct_context": correct_context,
            "correct_expected_answer": correct_expected_answer,
            "correct_keywords": correct_keywords,
            "fully_correct_number": len(fully_correct),
            "fully_correct": fully_correct,
            "mostly_correct_number": len(mostly_correct),
            "mostly_correct": mostly_correct,
            "partially_correct_number": len(partially_correct),
            "partially_correct": partially_correct,
            "incorrect_number": len(incorrect),
            "incorrect": incorrect
        }

        save_json(statistics, f"tests/results/{filename}.json")

    def _calculate_percentage_in_total_tests(self, count):
        total_tests = len(self.tests_results)
        return (count / total_tests * 100) if total_tests > 0 else 0
