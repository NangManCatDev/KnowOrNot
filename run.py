from vectorstore import vectorstore_main
from vectorstore_to_NEO import convert_vectorstore_to_NEO
from query_to_NEO import query_to_neo_query
# from neo_executor import execute_neo_query
# from response
import logging
import os

# Info: 로깅 설정
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Info: 최종수정: 2025-01-05, 수정자: NangManCat
# Note: 해당 코드는 NEO와의 상호작용을 통해 자연어 질문을 처리하고 결과를 반환하는 코드입니다.
# Note: 입력된 자연어 질문을 NEO 쿼리로 변환하여 "facts.nkb"에 정의된 규칙 기반으로 질의응답을 수행합니다.


# Info: 벡터 저장소가 존재하는지 확인
vectorstore_path = "vectorstore"
if os.path.exists(vectorstore_path):
    logging.info(f"✅ 벡터 저장소({vectorstore_path})가 이미 존재하므로 전처리 및 벡터 저장소 생성을 스킵합니다.")
else:
    # Info: 전처리에 필요한 모듈들을 여기서 import
    from ollama_preprocessor import process_document
    import data_preprocessor
    
    # Info: 문서 전처리 실행
    logging.info("🔄 문서 기본 전처리 시작...")
    sentences = process_document()
    logging.info(f"✅ 문서 기본 전처리 완료! {len(sentences)}개의 문장 추출")

    # Info: 추가 데이터 전처리 실행
    logging.info("🔄 데이터 상세 전처리 시작...")
    data_preprocessor.process_data()
    logging.info("✅ 데이터 상세 전처리 완료!")

    # Info: 벡터 저장소 생성
    logging.info("🔄 벡터 저장소 생성 중...")
    vectorstore_main()
    logging.info("✅ 벡터 저장소 생성 완료!")


# Info: NEO 변환 활성화 여부
# Note: True, False 설정 시 각각 활성화 또는 비활성화 됨
ENABLE_NEO_CONVERSION = True

# Info: NEO 변환 실행 (필요 시)
neo_facts_path = "NEO/facts.nkb"
if ENABLE_NEO_CONVERSION:
    if os.path.exists(neo_facts_path):
        logging.info(f"✅ NEO 파일({neo_facts_path})이 이미 존재하므로 NEO 변환을 스킵합니다.")
    else:
        logging.info("🔄 Vectorstore를 NEO 언어로 변환 중...")
        convert_vectorstore_to_NEO(
            vectorstore_path="vectorstore",  # Check: 벡터 저장소 디렉토리
            doc_txt_path="debug/ollama_preprocessed.txt",  # Check: RAG할 파일 경로
            system_prompt_path="NEO_system_prompt.txt",  # Check: system_prompt 파일 경로
        )
        logging.info("✅ NEO 변환 완료!")


# Info: NEO 질의응답 실행
if __name__ == "__main__":
    try:
        # Info: 사용자 질문 입력
        user_question = input("Enter your question: ")

        # Info: NEO nkb 파일 경로 설정
        kb_path = "NEO/facts.nkb"

        # Info: 자연어를 NEO 쿼리로 변환
        neo_query = query_to_neo_query(kb_path, user_question)
        print(f"NEO Query: {neo_query}")

        # # Info: NEO 쿼리 실행 및 결과 출력
        # result = execute_neo_query(kb_path, neo_query)
        # print("\nQuery Results:")
        # print(result)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
