import json
import os.path

from colorama import init

from CustomRag import CustomRag
from utils.log_utils import s_i

TEST_QUESTION = [
    "What is the main purpose of RFC6265?",
    "How does the Set-Cookie header work?",
    "What are the security considerations for cookies mentioned in RFC6265?",
    "What is the difference between session cookies and persistent cookies?",
    "How does the Domain attribute affect cookie scope?",
    "What is the purpose of the Path attribute in cookies?",
    "What does the Secure attribute do?",
    "How does the HttpOnly flag enhance cookie security?",
    "What are the restrictions on cookie size and number?",
    "How should user agents handle invalid cookies?",
    "What is the SameSite attribute and why was it introduced?",
    "How does cookie expiration work with the Expires and Max-Age attributes?",
    "What are the privacy concerns related to third-party cookies?",
    "How should servers set cookies to prevent CSRF attacks?",
    "What characters are allowed in cookie names and values?",
    "How do subdomains interact with cookie Domain attributes?",
    "What happens when multiple cookies have the same name?",
    "How should cookies be transmitted in HTTP requests?",
    "What are the recommendations for cookie deletion?",
    "How does RFC6265 differ from previous cookie specifications?",
]

if __name__ == "__main__":
    init(autoreset=True)
    rag = CustomRag()
    rag.clear_vectorstore()
    rag.load_pdf_files(use_semantic=True)
    print(f"\n{s_i} Total Questions: ", len(TEST_QUESTION))
    q_a = {}
    for index, question in enumerate(TEST_QUESTION):
        print(f"{s_i} Processing Question {index + 1}/{len(TEST_QUESTION)}:")
        print(f"Question: {question}")
        answer = rag.ask(question)
        print(f"Answer: {answer}")
        q_a[index + 1] = {
            "question": question,
            "answer": answer,
        }

    if not os.path.exists("./tests_results"):
        os.makedirs("./tests_results")
    with open("tests_results/rfc6265_test_results_1.json", "w", encoding="utf-8") as f:
        json.dump(q_a, f, indent=4, ensure_ascii=False)

    print(f"{s_i} Test Completed. Total Questions Answered: {len(q_a)}")

