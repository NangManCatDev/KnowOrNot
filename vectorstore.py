import logging
import os
import json
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Info: 로깅 설정
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

def vectorstore_main():
    logging.debug("vectorstore 확인 중...")

    # Info: JSON 파일 경로
    json_path = "debug/data_preprocessed.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 파일 {json_path}을 찾을 수 없습니다.")

    # Info: JSON 로드 및 텍스트 추출
    loader = JSONLoader(
        file_path=json_path,
        jq_schema='.[]',  # JSON 배열의 각 항목을 처리
        text_content=False
    )
    documents = loader.load()

    # Info: 텍스트 분리
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)

    # Info: 임베딩 생성
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Info: 벡터 저장소 초기화 및 저장
    vectorstore_path = "vectorstore"
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=vectorstore_path
    )
    vectorstore.persist()
    print("vectorstore 생성 및 유지 중...")
