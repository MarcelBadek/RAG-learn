import os
import re

from colorama import Fore
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from utils.log_utils import s_i, s_l, s_s
from utils.utils import load_files
from llm_vendors import ask_ollama


DEFAULT_MODEL = "llama3.1"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
DEFAULT_BASE_URL = "localhost:11434"


class CustomRag:
    def __init__(self,
                 embedding_model=OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL, base_url=DEFAULT_BASE_URL),
                 split_chunk_size=500,
                 split_chunk_overlap=100,
                 persist_directory="./chroma_db"):
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        self.vectorstore = Chroma(
            collection_name="my_collection",
            embedding_function=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
            }
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=split_chunk_size,
            chunk_overlap=split_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
            strip_whitespace=True
        )

    def load_universe_files(self, path="documents/universe"):
        self._load_documents(
            path=path,
            file_type="txt",
            doc_type="universe",
            extractor=self._extract_text_from_txt
        )

    def load_rfc_files(self,  path="documents/rfc"):
        self._load_documents(
            path=path,
            file_type="pdf",
            doc_type="RFC",
            extractor=self._extract_text_from_pdf
        )

    def _load_documents(self, path, file_type, doc_type, extractor):
        print(f"{s_l} Loading {doc_type} documents...")
        files = load_files(path, file_type)
        print(f"{s_i} Found {len(files)} files in the documents directory")

        all_chunks = []
        for file in files:
            chunks = extractor(file)
            all_chunks.extend(chunks)

        print(f"{s_i} Created {len(all_chunks)} text chunks")
        print(f"{s_l} Adding documents to vector store...")
        self.vectorstore.add_documents(all_chunks)

    def _extract_text_from_txt(self, file):
        text = file.read_text(encoding="utf-8")
        return self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": file.name}]
        )

    def _extract_text_from_pdf(self, file):
        chunks = []
        pdf_reader = PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            text = re.sub(r'\n+', '\n', text)
            page_chunks = self.text_splitter.create_documents(
                [text],
                metadatas=[{"source": f"{file.name} - page {page_num}"}]
            )
            chunks.extend(page_chunks)
        return chunks

    def clear_vectorstore(self):
        print(f"{s_l} Clearing vector store...")
        self.vectorstore.delete_collection()
        print(f"{s_s} Vector store cleared.")

    def ask(self, question):
        documents = self._find_relevant_documents(question)
        print(f"{s_i} Founded documents:")
        for doc in documents:
            print(f"{Fore.CYAN + " " * 6} - {doc.replace("\n", " ")}")
        print(f"{s_l} Preparing context for LLM...")
        conceited_documents = "\n\n".join(documents)
        print(f"{s_l} Generating answer with LLM...")
        answer = ask_ollama(conceited_documents, question, model=DEFAULT_MODEL)
        return answer

    def _find_relevant_documents(self, question):
        print(f"{s_l} Retrieving documents for query: '{question}'...")
        documents = self.retriever.invoke(question)
        print(f"{s_i} Retrieved {len(documents)} relevant documents")
        return [doc.page_content for doc in documents]
