from enum import Enum
from typing import Optional, List

from colorama import Fore, Style


class LoggerCategory(Enum):
    LOADING = "LOADING"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    FULL_PROMPT = "FULL_PROMPT"
    STATISTICS = "STATISTICS"
    DOCUMENTS = "DOCUMENTS"
    PROCESSING_QUESTION = "PROCESSING_QUESTION"
    FULL_RESPONSE = "FULL_RESPONSE"


class CustomLogger:
    _instance: Optional['CustomLogger'] = None
    _configured = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._configured:
            self.categories = [
                LoggerCategory.LOADING,
                LoggerCategory.INFO,
                LoggerCategory.SUCCESS,
                LoggerCategory.ERROR,
                LoggerCategory.QUESTION,
                LoggerCategory.ANSWER,
                LoggerCategory.STATISTICS,
                LoggerCategory.DOCUMENTS,
                LoggerCategory.PROCESSING_QUESTION
            ]

    @classmethod
    def configure(cls, categories: Optional[List[LoggerCategory]] = None):
        instance = cls()
        if categories is not None:
            instance.categories = categories
        instance._configured = True

        if instance.categories:
            categories_str = ", ".join(cat.value for cat in instance.categories)
            print(
                f"{Fore.CYAN}[CONFIG] Logger configured with categories: [{categories_str}]{Style.RESET_ALL}"
            )

    @staticmethod
    def always(message: str):
        print(message)

    def loading(self, message: str):
        if LoggerCategory.LOADING in self.categories:
            print(f"{Fore.CYAN}[*] {message}...{Style.RESET_ALL}")

    def success(self, message: str):
        if LoggerCategory.SUCCESS in self.categories:
            print(f"{Fore.GREEN}[✓] {message}{Style.RESET_ALL}")

    def info(self, message: str):
        if LoggerCategory.INFO in self.categories:
            print(f"[i] {message}")

    def error(self, message: str):
        if LoggerCategory.ERROR in self.categories:
            print(f"{Fore.RED}[✗] {message}{Style.RESET_ALL}")

    def question(self, message: str):
        if LoggerCategory.QUESTION in self.categories:
            print(f"{Fore.LIGHTCYAN_EX}[?] {message}{Style.RESET_ALL}")

    def answer(self, message: str):
        if LoggerCategory.ANSWER in self.categories:
            print(f"{Fore.MAGENTA}RAG: {message}{Style.RESET_ALL}")

    def full_prompt(self, message: str):
        if LoggerCategory.FULL_PROMPT in self.categories:
            print(f"{Fore.LIGHTYELLOW_EX}{message}{Style.RESET_ALL}")

    def statistics(self, message: str):
        if LoggerCategory.STATISTICS in self.categories:
            print(f"{Fore.YELLOW}[s] {message}{Style.RESET_ALL}")

    def documents(self, message: str):
        if LoggerCategory.DOCUMENTS in self.categories:
            print(f"{Fore.BLUE}[d] {message}{Style.RESET_ALL}")

    def processing_question(self, message: str):
        if LoggerCategory.PROCESSING_QUESTION in self.categories:
            print(f"{Fore.LIGHTBLUE_EX}[p] {message}{Style.RESET_ALL}")

    def full_response(self, message: str):
        if LoggerCategory.FULL_RESPONSE in self.categories:
            print(f"{Fore.LIGHTGREEN_EX}[r] {message}{Style.RESET_ALL}")

log = CustomLogger()
