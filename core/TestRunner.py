import os

from colorama import Fore

from core.AdjustedOllama import AdjustedOllama
from utils.CustomLogger import log
from utils.utils import save_json, get_current_datetime


class TestRunner:
    def __init__(self, rag_instance, model):
        self.rag = rag_instance
        self.adjusted_model = AdjustedOllama(model)
        self.tests_results = {}

        if not os.path.exists("../tests/results"):
            os.makedirs("../tests/results")

    def run_test(self, details):
        question, expected_answer, keywords = self._separate_question(details)

        log.processing_question(
            f"Question: {question}, Expected Answer: {expected_answer}, Keywords: {keywords}")

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
        statistics = []
        for i in range(run_number):
            log.always(f"Running test set iteration {i + 1} of {run_number}...")
            self.tests_results = {}
            for test in test_set:
                self.run_test(test)
            statistics.append(self.generate_summary(save_summary=False, show_summary=False))

        total_tests = 0
        total_correct_context = 0
        total_correct_expected_answer = 0
        total_correct_keywords = 0
        total_avg_response_time = 0
        total_avg_token_usage = 0
        total_fully_correct = 0
        total_mostly_correct = 0
        total_partially_correct = 0
        total_incorrect = 0

        for stat in statistics:
            total_tests = total_tests + stat.get("total_tests", 0)
            total_correct_context = total_correct_context + stat.get("correct_context", 0)
            total_correct_expected_answer = total_correct_expected_answer + stat.get("correct_expected_answer", 0)
            total_correct_keywords = total_correct_keywords + stat.get("correct_keywords", 0)
            total_avg_response_time = total_avg_response_time + stat.get("response_time_average", 0)
            total_avg_token_usage = total_avg_token_usage + stat.get("average_prompt_tokens_per_test", 0)
            total_fully_correct = total_fully_correct + stat.get("fully_correct_number", 0)
            total_mostly_correct = total_mostly_correct + stat.get("mostly_correct_number", 0)
            total_partially_correct = total_partially_correct + stat.get("partially_correct_number", 0)
            total_incorrect = total_incorrect + stat.get("incorrect_number", 0)
        log.always(f"After {run_number} runs of the test set:")
        log.always(f"Total tests: {total_tests}")
        log.always(
            f"Average correct based on context: {Fore.LIGHTBLUE_EX}{total_correct_context / run_number:.2f} ({(total_correct_context / total_tests * 100):.2f}%)")
        log.always(
            f"Average correct based on expected Answer: {Fore.LIGHTBLUE_EX}{total_correct_expected_answer / run_number:.2f} ({(total_correct_expected_answer / total_tests * 100):.2f}%)")
        log.always(
            f"Average correct based on keywords: {Fore.LIGHTBLUE_EX}{total_correct_keywords / run_number:.2f} ({(total_correct_keywords / total_tests * 100):.2f}%)")

        stats_summary = {
            "total_runs": run_number,
            "total_tests": total_tests,
            "total_correct_context": total_correct_context,
            "total_correct_expected_answer": total_correct_expected_answer,
            "total_correct_keywords": total_correct_keywords,
            "average_correct_context": total_correct_context / run_number,
            "average_correct_expected_answer": total_correct_expected_answer / run_number,
            "average_correct_keywords": total_correct_keywords / run_number,
            "average_response_time": total_avg_response_time / run_number,
            "average_token_usage": total_avg_token_usage / run_number,
            "average_fully_correct": total_fully_correct / run_number,
            "average_mostly_correct": total_mostly_correct / run_number,
            "average_partially_correct": total_partially_correct / run_number,
            "average_incorrect": total_incorrect / run_number,
        }

        self.generate_multirun_statistics_per_question(statistics)
        save_json(stats_summary, f"tests/results/multirun_test_summary_{get_current_datetime()}.json")

    def save_tests_results(self, base_filename="test_results"):
        current_date_time = get_current_datetime()
        filename_details = f"{base_filename}_{current_date_time}_details.json"
        filename_simple = f"{base_filename}_{current_date_time}_qa.json"
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

        self.generate_summary(filename_statistics)

    def generate_summary(self, filename=None, show_summary=True, save_summary=True):
        if filename is None:
            filename = f"test_summary_{get_current_datetime()}.json"
        fully_correct = []
        mostly_correct = []
        partially_correct = []
        incorrect = []
        response_time = 0
        token_usage = 0
        total_tests = len(self.tests_results)
        model_name = self.tests_results.get(1)["details"].get("model", "unknown") if total_tests > 0 else "unknown"
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

        if show_summary:
            log.always(f"Total tests: {total_tests}")
            log.always(f"Average response time: {Fore.LIGHTBLUE_EX}{average_response_time:.2f} seconds")
            log.always(f"Total prompt tokens used: {Fore.LIGHTBLUE_EX}{token_usage}")
            log.always(f"Average prompt tokens per test: {Fore.LIGHTBLUE_EX}{average_token_usage:.2f}")
            log.always(
                f"Correct based on context: {Fore.LIGHTBLUE_EX}{correct_context} ({self._calculate_percentage_in_total_tests(correct_context):.2f}%)")
            log.always(
                f"Correct based on expected answer: {Fore.LIGHTBLUE_EX}{correct_expected_answer} ({self._calculate_percentage_in_total_tests(correct_expected_answer):.2f}%)")
            log.always(
                f"Correct based on keywords: {Fore.LIGHTBLUE_EX}{correct_keywords} ({self._calculate_percentage_in_total_tests(correct_keywords):.2f}%)")
            log.always(
                f"Fully correct (correct 3/3): {Fore.LIGHTBLUE_EX}{len(fully_correct)} ({self._calculate_percentage_in_total_tests(len(fully_correct)):.2f}%)")
            log.always(
                f"Mostly correct (correct 2/3): {Fore.LIGHTBLUE_EX}{len(mostly_correct)} ({self._calculate_percentage_in_total_tests(len(mostly_correct)):.2f}%)")
            log.always(
                f"Partially correct (correct 1/3): {Fore.LIGHTBLUE_EX}{len(partially_correct)} ({self._calculate_percentage_in_total_tests(len(partially_correct)):.2f}%)")
            log.always(
                f"Incorrect (correct 0/3): {Fore.LIGHTBLUE_EX}{len(incorrect)} ({self._calculate_percentage_in_total_tests(len(incorrect)):.2f}%)")

        statistics = {
            "model": model_name,
            "total_tests": total_tests,
            "total_response_time": response_time,
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

        if save_summary:
            save_json(statistics, f"tests/results/{filename}")

        return statistics

    def generate_multirun_statistics_per_question(self, statistics):
        parsed_statistics = {}

        all_questions = set()
        for stat in statistics:
            all_questions.update(stat.get("fully_correct", []))
            all_questions.update(stat.get("mostly_correct", []))
            all_questions.update(stat.get("partially_correct", []))
            all_questions.update(stat.get("incorrect", []))

        total_runs = len(statistics)

        for question in all_questions:
            fully_correct_count = sum(1 for stat in statistics if question in stat.get("fully_correct", []))
            mostly_correct_count = sum(1 for stat in statistics if question in stat.get("mostly_correct", []))
            partially_correct_count = sum(1 for stat in statistics if question in stat.get("partially_correct", []))
            incorrect_count = sum(1 for stat in statistics if question in stat.get("incorrect", []))

            success_count = fully_correct_count + mostly_correct_count

            category_counts = [fully_correct_count, mostly_correct_count, partially_correct_count, incorrect_count]
            max_category = max(category_counts)
            stability_score = max_category / total_runs if total_runs > 0 else 0

            categories = ["fully_correct", "mostly_correct", "partially_correct", "incorrect"]
            dominant_category = categories[category_counts.index(max_category)]

            parsed_statistics[question] = {
                "total_runs": total_runs,
                "fully_correct_count": fully_correct_count,
                "mostly_correct_count": mostly_correct_count,
                "partially_correct_count": partially_correct_count,
                "incorrect_count": incorrect_count,
                "fully_correct_percentage": self._calculate_percentage_in_total_tests(fully_correct_count, total_runs),
                "mostly_correct_percentage": self._calculate_percentage_in_total_tests(mostly_correct_count, total_runs),
                "partially_correct_percentage": self._calculate_percentage_in_total_tests(partially_correct_count, total_runs),
                "incorrect_percentage": self._calculate_percentage_in_total_tests(incorrect_count, total_runs),
                "success_rate": self._calculate_percentage_in_total_tests(success_count, total_runs),
                "stability_score": stability_score,
                "dominant_category": dominant_category,
                "is_stable": stability_score >= 0.8,
                "is_problematic": incorrect_count > (total_runs * 0.80)
            }

        # Add summary section
        summary = {
            "total_questions": len(all_questions),
            "total_runs": total_runs,
            "stable_questions": sum(1 for q in parsed_statistics.values() if q["is_stable"]),
            "problematic_questions": sum(1 for q in parsed_statistics.values() if q["is_problematic"]),
            "average_success_rate": sum(q["success_rate"] for q in parsed_statistics.values()) / len(parsed_statistics) if parsed_statistics else 0
        }

        output = {
            "summary": summary,
            "per_question_statistics": parsed_statistics
        }

        save_json(output, f"tests/results/multirun_test_summary_{get_current_datetime()}_per_question.json")

    def _calculate_percentage_in_total_tests(self, count, total_tests=None):
        if total_tests is None:
            total_tests = len(self.tests_results)
        return (count / total_tests * 100) if total_tests > 0 else 0

    @staticmethod
    def _separate_question(details):
        question = details.get("question", "")
        expected_answer = details.get("expected_answer", "")
        keywords = details.get("keywords", [])

        return question, expected_answer, keywords
