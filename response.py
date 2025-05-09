import logging
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Note: 해당 모듈은 NEO_executor.py에서 결과가 나오면 작성토록 함.

# Info: 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def init_llm() -> Ollama:
    """Ollama LLM 초기화"""
    try:
        return Ollama(model="deepseek-r1:14b")
    except Exception as e:
        logging.error(f"LLM 초기화 중 오류 발생: {e}")
        raise
