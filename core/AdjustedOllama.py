from langchain_core.outputs import LLMResult
from langchain_ollama import OllamaLLM

from utils.CustomLogger import log

ASK_INTRO = """You are an AI assistant whose answers must rely exclusively on the context supplied. Follow these rules strictly:

1. Use only the provided context as your knowledge source.
Do not use outside knowledge, assumptions, or general domain understanding.

2. Keep answers accurate, concise, and to the point.

3. Answer using provided context as it would be your only source of truth.

4. You must NOT mention relation of your answer to context. Context is your knowledge so answers should reflect that it is your own knowledge.
Do NOT use phrases such as:
- “based on the provided context”
- “the context says”
- “according to the context”
- or any similar wording.

5. If the context does not contain enough information to answer the question, respond exactly with:
“I don't have sufficient information to answer this question.”
Do NOT add explanations, alternative phrasing, disclaimers, or references to the context."""

PROMPT_VALIDATION_CONTEXT = """You are an AI language model developed to validate answers correctness of provided answers based on the provided context. 
1. If the context does not contain sufficient information to validate the answer or the answer is not correct, respond with 'Incorrect'. 
2. If the answer is correct, respond with 'Correct'.
3. If the answer is “I don't have sufficient information to answer this question.” or similar, respond with 'Correct'."""

PROMPT_VALIDATION_EXPECTED_ANSWER = """You are an AI language model developed to validate answers correctness of provided answers based on the provided expected answer. 
1. If the expected answer does not contain sufficient information to validate the answer or the answer is not correct, respond with 'Incorrect'. 
2. If the answer is correct, respond with 'Correct'."""

PROMPT_VALIDATION_EXPECTED_KEYWORDS = """You are an AI language model developed to validate answers correctness of provided answers based on the provided expected keywords. 
1. If the expected keywords do not contain sufficient information to validate the answer or the answer is not correct, respond with 'Incorrect'. 
2. If the answer is correct, respond with 'Correct'."""

ASK_TEMPLATE = "{intro} \n\nContext: {context} \n\nQuestion: {question} \n\nYour answer: "
VALIDATION_TEMPLATE = "{intro} \n\n{validation_context} \n\nAnswer: {answer} \n\nYour validation (Correct/Incorrect): "


class AdjustedOllama:
    def __init__(self, model):
        self.llm = OllamaLLM(model=model, temperature=0.1)
        self.validation_llm = OllamaLLM(model=model, temperature=0.0)

    def ask_ollama(self, context: str, prompt: str):
        contents = ASK_TEMPLATE.format(
            intro=ASK_INTRO,
            context=context,
            question=prompt
        )
        response_text, result = self.send_prompt_to_ollama(contents, validation=False)

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
            log.statistics(
                f"Model: {details['model']}, Prompt Tokens: {details['prompt_eval_count']}, Response Tokens: {details['eval_count']}, Duration: {details['total_duration_s']}s")

        return response_text, details

    def validate_answer_with_context(self, answer: str, context: str):
        contents = VALIDATION_TEMPLATE.format(
            intro=PROMPT_VALIDATION_CONTEXT,
            validation_context=f"Context: {context}",
            answer=answer
        )
        response_text, _ = self.send_prompt_to_ollama(contents, validation=True)

        return self._interpret_validation_response(response_text)

    def validate_answer_with_expected_answer(self, answer: str, expected_answer: str):
        contents = VALIDATION_TEMPLATE.format(
            intro=PROMPT_VALIDATION_EXPECTED_ANSWER,
            validation_context=f"Expected Answer: {expected_answer}",
            answer=answer
        )
        response_text, _ = self.send_prompt_to_ollama(contents, validation=True)

        return self._interpret_validation_response(response_text)

    def validate_answer_with_expected_keywords(self, answer: str, expected_keywords: str):
        contents = VALIDATION_TEMPLATE.format(
            intro=PROMPT_VALIDATION_EXPECTED_KEYWORDS,
            validation_context=f"Expected Keywords: {expected_keywords}",
            answer=answer
        )
        response_text, _ = self.send_prompt_to_ollama(contents, validation=True)

        return self._interpret_validation_response(response_text)

    def send_prompt_to_ollama(self, prompt: str, validation: bool = False):
        log.full_prompt(prompt)
        if validation:
            result: LLMResult = self.validation_llm.generate([prompt])
        else:
            result: LLMResult = self.llm.generate([prompt])
        response_text = result.generations[0][0].text.strip()

        log.full_response(response_text)

        return response_text, result

    @staticmethod
    def _interpret_validation_response(response: str) -> bool:
        response_lower = response.lower().strip()
        return "correct" in response_lower and "incorrect" not in response_lower
