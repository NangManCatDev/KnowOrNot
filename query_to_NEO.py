import logging
from typing import Optional
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# FIXME: 프롬프트 개선이 필요하다. 현재는 디버그 용도의 프롬프트가 사용되고 있다.

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

def load_kb_content(kb_path: str) -> str:
    """NEO KB 파일 로드"""
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"KB 파일 로드 중 오류 발생: {e}")
        raise

def create_vectorstore(text: str) -> FAISS:
    """벡터 저장소 생성"""
    try:
        # Info: 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Info: 임베딩 및 벡터 저장소 생성
        embeddings = OllamaEmbeddings(model="llama2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        return vectorstore
    except Exception as e:
        logging.error(f"벡터 저장소 생성 중 오류 발생: {e}")
        raise

def query_to_neo_query(kb_path: str, user_question: str) -> Optional[str]:
    """
    자연어 질문을 NEO 쿼리로 변환
    
    Args:
        kb_path (str): NEO KB 파일 경로
        user_question (str): 사용자의 자연어 질문
    
    Returns:
        str: 변환된 NEO 쿼리
    """
    try:
        # Info: KB 파일 로드
        kb_content = load_kb_content(kb_path)
        
        # Info: 벡터 저장소 생성
        vectorstore = create_vectorstore(kb_content)
        
        # Info: 관련 컨텍스트 검색
        relevant_chunks = vectorstore.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in relevant_chunks])
        
        # Info: 시스템 프롬프트 로드
        system_prompt_path = "NEO_query_system_prompt.txt"
        if not os.path.exists(system_prompt_path):
            raise FileNotFoundError(f"시스템 프롬프트 파일을 찾을 수 없습니다: {system_prompt_path}")
            
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        # Info: LLM 초기화
        llm = init_llm()
        
        # Info: 프롬프트 구성
        prompt = f"""
        시스템 프롬프트:
        {system_prompt}
        
        NEO KB 컨텍스트:
        {context}
        
        사용자 질문:
        {user_question}
        
        위 질문을 NEO 쿼리 형식으로 변환해주세요.
        """
        
        # Info: LLM을 통한 쿼리 변환
        neo_query = llm.invoke(prompt).strip()
        
        logging.info(f"자연어 질문이 NEO 쿼리로 변환되었습니다: {neo_query}")
        
        # Info: 디버그 디렉토리 생성
        os.makedirs("debug", exist_ok=True)
        
        # Info: 디버그 정보를 파일로 저장
        debug_info = f"""원본 질문: {user_question}
컨텍스트:
{context}

변환된 NEO 쿼리:
{neo_query}
"""
        
        with open("debug/query_to_NEO.txt", "w", encoding="utf-8") as f:
            f.write(debug_info)
            
        logging.info("디버그 정보가 debug/query_to_NEO.txt에 저장되었습니다.")
        
        return neo_query
        
    except Exception as e:
        logging.error(f"쿼리 변환 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    # Info: 테스트 예제
    test_question = "이순신의 사망일은 언제인가요?"
    kb_path = "NEO/facts.nkb"
    result = query_to_neo_query(kb_path, test_question)
    if result:
        print(f"변환된 NEO 쿼리: {result}")
