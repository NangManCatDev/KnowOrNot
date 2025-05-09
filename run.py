from vectorstore import vectorstore_main
from vectorstore_to_NEO import convert_vectorstore_to_NEO
from query_to_NEO import query_to_neo_query
# from neo_executor import execute_neo_query
# from response
import logging
import os
import gradio as gr
import time
import io
import sys

# 텍스트 로그를 저장할 버퍼
log_buffer = []
MAX_LOG_ENTRIES = 500  # 최대 로그 항목 수

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 커스텀 로그 핸들러 정의
class LogCapture(logging.Handler):
    def emit(self, record):
        log_msg = self.format(record)
        log_buffer.append(log_msg)
        if len(log_buffer) > MAX_LOG_ENTRIES:
            log_buffer.pop(0)
        # 원래 stdout에도 출력
        print(log_msg)

# 로그 핸들러 추가
handler = LogCapture()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(handler)

# Info: 최종수정: 2025-01-05, 수정자: NangManCat
# Note: 해당 코드는 NEO와의 상호작용을 통해 자연어 질문을 처리하고 결과를 반환하는 코드입니다.
# Note: 입력된 자연어 질문을 NEO 쿼리로 변환하여 "facts.nkb"에 정의된 규칙 기반으로 질의응답을 수행합니다.


def initialize_system():
    """시스템 초기화 함수: 벡터 저장소 및 NEO 변환을 확인하고 필요시 생성"""
    log_messages = []
    
    # Info: 벡터 저장소가 존재하는지 확인
    vectorstore_path = "vectorstore"
    if os.path.exists(vectorstore_path):
        message = f"✅ 벡터 저장소({vectorstore_path})가 이미 존재하므로 전처리 및 벡터 저장소 생성을 스킵합니다."
        logging.info(message)
        log_messages.append(message)
    else:
        # Info: 전처리에 필요한 모듈들을 여기서 import
        from ollama_preprocessor import process_document
        import data_preprocessor
        
        # Info: 문서 전처리 실행
        message = "🔄 문서 기본 전처리 시작..."
        logging.info(message)
        log_messages.append(message)
        
        sentences = process_document()
        message = f"✅ 문서 기본 전처리 완료! {len(sentences)}개의 문장 추출"
        logging.info(message)
        log_messages.append(message)

        # Info: 추가 데이터 전처리 실행
        message = "🔄 데이터 상세 전처리 시작..."
        logging.info(message)
        log_messages.append(message)
        
        data_preprocessor.process_data()
        message = "✅ 데이터 상세 전처리 완료!"
        logging.info(message)
        log_messages.append(message)

        # Info: 벡터 저장소 생성
        message = "🔄 벡터 저장소 생성 중..."
        logging.info(message)
        log_messages.append(message)
        
        vectorstore_main()
        message = "✅ 벡터 저장소 생성 완료!"
        logging.info(message)
        log_messages.append(message)

    # Info: NEO 변환 활성화 여부
    # Note: True, False 설정 시 각각 활성화 또는 비활성화 됨
    ENABLE_NEO_CONVERSION = True

    # Info: NEO 변환 실행 (필요 시)
    neo_facts_path = "NEO/facts.nkb"
    if ENABLE_NEO_CONVERSION:
        if os.path.exists(neo_facts_path):
            message = f"✅ NEO 파일({neo_facts_path})이 이미 존재하므로 NEO 변환을 스킵합니다."
            logging.info(message)
            log_messages.append(message)
        else:
            message = "🔄 Vectorstore를 NEO 언어로 변환 중..."
            logging.info(message)
            log_messages.append(message)
            
            convert_vectorstore_to_NEO(
                vectorstore_path="vectorstore",  # Check: 벡터 저장소 디렉토리
                doc_txt_path="debug/ollama_preprocessed.txt",  # Check: RAG할 파일 경로
                system_prompt_path="NEO_system_prompt.txt",  # Check: system_prompt 파일 경로
            )
            message = "✅ NEO 변환 완료!"
            logging.info(message)
            log_messages.append(message)
    
    return "\n".join(log_messages)


def process_question(question):
    """사용자 질문을 처리하는 함수"""
    try:
        # Info: NEO nkb 파일 경로 설정
        kb_path = "NEO/facts.nkb"

        # Info: 자연어를 NEO 쿼리로 변환
        neo_query = query_to_neo_query(kb_path, question)
        
        # # Info: NEO 쿼리 실행 및 결과 출력 (주석 처리되어 있음)
        # result = execute_neo_query(kb_path, neo_query)
        # return f"NEO 쿼리: {neo_query}\n\n쿼리 결과:\n{result}"
        
        return f"NEO 쿼리: {neo_query}"
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return f"오류 발생: {str(e)}"


def get_terminal_logs():
    """현재까지 수집된 로그를 반환"""
    return "\n".join(log_buffer)


# Gradio 인터페이스 설정
def create_interface():
    """Gradio 인터페이스 생성 함수"""
    with gr.Blocks(title="KnowOrNot - NEO 질의응답 시스템") as interface:
        gr.Markdown("# KnowOrNot - NEO 질의응답 시스템")
        
        # 시스템 초기화 상태
        system_initialized = gr.State(False)
        
        # 시작 화면: 시작 버튼만 표시
        with gr.Group(visible=True) as start_screen:
            gr.Markdown("## KnowOrNot 시스템을 시작하려면 아래 버튼을 클릭하세요")
            start_button = gr.Button("시스템 시작", variant="primary", scale=1, size="lg")
            start_status = gr.Textbox(label="상태", value="시스템이 준비되었습니다. 시작 버튼을 눌러주세요.", interactive=False)
        
        # 메인 화면: 시스템 초기화 후 표시되는 화면
        with gr.Group(visible=False) as main_screen:
            with gr.Tabs():
                with gr.TabItem("시스템 상태"):
                    system_log = gr.Textbox(label="시스템 로그", lines=10, interactive=False)
                    refresh_button = gr.Button("상태 새로고침")
                
                with gr.TabItem("터미널 로그"):
                    gr.Markdown("### 터미널 출력")
                    terminal_log = gr.Textbox(label="터미널 출력", lines=20, autoscroll=True, interactive=False)
                    with gr.Row():
                        clear_logs_button = gr.Button("로그 지우기")
                        manual_refresh_button = gr.Button("로그 새로고침")
                    
                    # 로그 지우기 함수
                    def clear_logs():
                        global log_buffer
                        log_buffer = []
                        return ""
                    
                    # 로그 지우기 버튼 클릭 시 로그 지우기
                    clear_logs_button.click(
                        fn=clear_logs,
                        outputs=[terminal_log]
                    )
                    
                    # 수동 새로고침 버튼 클릭 시 현재 로그 전체 가져오기
                    manual_refresh_button.click(
                        fn=get_terminal_logs,
                        outputs=[terminal_log]
                    )
                    
                with gr.TabItem("질의응답"):
                    question_input = gr.Textbox(label="질문 입력", placeholder="여기에 질문을 입력하세요...")
                    submit_button = gr.Button("질문 제출", variant="primary")
                    answer_output = gr.Textbox(label="응답 결과", lines=10, interactive=False)
        
        # 시작 버튼 클릭 시 시스템 초기화 및 화면 전환
        def on_start_click():
            log = initialize_system()
            all_logs = get_terminal_logs()
            return {
                start_screen: gr.update(visible=False),
                main_screen: gr.update(visible=True),
                system_log: log,
                terminal_log: all_logs,
                system_initialized: True
            }
        
        start_button.click(
            fn=on_start_click,
            outputs=[start_screen, main_screen, system_log, terminal_log, system_initialized]
        )
        
        # 상태 새로고침 버튼 클릭 시
        refresh_button.click(
            fn=lambda: initialize_system(),
            outputs=[system_log]
        )
        
        # 질문 제출 버튼 클릭 시
        submit_button.click(
            fn=process_question,
            inputs=[question_input],
            outputs=[answer_output]
        )
    
    return interface


# Info: 메인 함수
if __name__ == "__main__":
    # 시작 메시지 출력
    logging.info("KnowOrNot - NEO 질의응답 시스템을 시작합니다.")
    
    # Gradio 인터페이스 실행
    interface = create_interface()
    interface.launch(share=True)
