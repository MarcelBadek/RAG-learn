from colorama import Fore
from langchain_core.outputs import LLMResult
from langchain_ollama import OllamaLLM

from utils.log_utils import s_i

ASK_INTRO = "You are an AI language model developed to provide accurate and concise answers based on the provided context. Always refer to the context when formulating your responses. If the context does not contain sufficient information to answer the question, respond with 'I don't know'. Avoid fabricating information."

PROMPT_VALIDATION_CONTEXT = "You are an AI language model developed to validate answers correctness of provided answers based on the provided context. If the context does not contain sufficient information to validate the answer or the answer is not correct, respond with 'Incorrect'. If the answer is correct, respond with 'Correct'."
PROMPT_VALIDATION_EXPECTED_ANSWER = "You are an AI language model developed to validate answers correctness of provided answers based on the provided expected answer. If the expected answer does not contain sufficient information to validate the answer or the answer is not correct, respond with 'Incorrect'. If the answer is correct, respond with 'Correct'."
PROMPT_VALIDATION_EXPECTED_KEYWORDS = "You are an AI language model developed to validate answers correctness of provided answers based on the provided expected keywords. If the expected keywords do not contain sufficient information to validate the answer or the answer is not correct, respond with 'Incorrect'. If the answer is correct, respond with 'Correct'."

ASK_TEMPLATE = "{intro} \nContext: {context} \nQuestion: {question} \nYour answer: "
VALIDATION_TEMPLATE = "{intro} \n{validation_context} \nAnswer: {answer} \nYour validation (Correct/Incorrect): "


class AdjustedOllama:
    def __init__(self, model, debug_print=False):
        self.llm = OllamaLLM(model=model)
        self.print_details = debug_print

    def ask_ollama(self, context: str, prompt: str):
        contents = ASK_TEMPLATE.format(
            intro=ASK_INTRO,
            context=context,
            question=prompt
        )
        response_text, result = self.send_prompt_to_ollama(contents)

        details = None
        info = result.generations[0][0].generation_info
        if info:
            details = {
                'model': info.get('model'),
                'prompt_eval_count': info.get('prompt_eval_count'),
                'eval_count': info.get('eval_count'),
                'total_duration_s': f"{(info.get('total_duration') / 1_000_000_000):.2f}" if info.get(
                    'total_duration') else "N/A"
            }
            print(
                f"{Fore.YELLOW + s_i} Stats | Model: {details['model']}, Prompt Tokens: {details['prompt_eval_count']}, Response Tokens: {details['eval_count']}, Duration: {details['total_duration_s']}s")

        return response_text, details

    def validate_answer_with_context(self, answer: str, context: str):
        contents = VALIDATION_TEMPLATE.format(
            intro=PROMPT_VALIDATION_CONTEXT,
            validation_context=f"Context: {context}",
            answer=answer
        )
        response_text, _ = self.send_prompt_to_ollama(contents)

        return self._interpret_validation_response(response_text)

    def validate_answer_with_expected_answer(self, answer: str, expected_answer: str):
        contents = VALIDATION_TEMPLATE.format(
            intro=PROMPT_VALIDATION_EXPECTED_ANSWER,
            validation_context=f"Expected Answer: {expected_answer}",
            answer=answer
        )
        response_text, _ = self.send_prompt_to_ollama(contents)

        return self._interpret_validation_response(response_text)

    def validate_answer_with_expected_keywords(self, answer: str, expected_keywords: str):
        contents = VALIDATION_TEMPLATE.format(
            intro=PROMPT_VALIDATION_EXPECTED_KEYWORDS,
            validation_context=f"Expected Keywords: {expected_keywords}",
            answer=answer
        )
        response_text, _ = self.send_prompt_to_ollama(contents)

        return self._interpret_validation_response(response_text)

    def send_prompt_to_ollama(self, prompt: str):
        # print(Fore.RED + prompt)
        result: LLMResult = self.llm.generate([prompt])
        response_text = result.generations[0][0].text.strip()

        if self.print_details:
            print(Fore.RED + s_i + response_text)

        return response_text, result

    @staticmethod
    def _interpret_validation_response(response: str) -> bool:
        response_lower = response.lower().strip()
        return "correct" in response_lower and "incorrect" not in response_lower
