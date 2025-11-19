import os

from colorama import Fore
from dotenv import load_dotenv
from google import genai
from langchain_core.outputs import LLMResult
from langchain_ollama import OllamaLLM
from openai import OpenAI

from utils.log_utils import s_i


INTRO = "Use the following context to answer the question. If you don't know the answer, say that you don't know and do not try to make up an answer.\n\nContext:\n"


def ask_open_ai(context: str, prompt: str, model: str = "gpt-3.5-turbo") -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(
        api_key=api_key,
    )
    response = client.responses.create(
        model=model,
        instructions=context,
        input=prompt
    )
    client.close()

    return response.output_text

def ask_gemini(context: str, prompt: str, model: str = "gemini-2.5-flash") -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    contents = f'{INTRO} {context}, Question: {prompt}, Your answer: '
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents
    )
    client.close()

    return response.text

def ask_ollama(context: str, prompt: str, model: str = "llama3.2", step: str = None):
    contents = f'{INTRO} {context}, Question: {prompt}, Your answer: '

    llm = OllamaLLM(model=model)
    result: LLMResult = llm.generate([contents])
    response_text = result.generations[0][0].text

    info = result.generations[0][0].generation_info
    if info:
        model_name = info.get('model', 'N/A')
        prompt_tokens = info.get('prompt_eval_count', 'N/A')
        response_tokens = info.get('eval_count', 'N/A')
        total_duration_ns = info.get('total_duration')
        duration_s = f"{(total_duration_ns / 1_000_000_000):.2f}s" if total_duration_ns else "N/A"
        print(
            f"{Fore.YELLOW + s_i} Stats | Model: {model_name}, Prompt Tokens: {prompt_tokens}, Response Tokens: {response_tokens}, Duration: {duration_s}")

    return response_text
