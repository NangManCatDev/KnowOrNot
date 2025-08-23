import os
import time
import json
import argparse
import glob
import subprocess
import requests
import threading
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import matplotlib
import platform

if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

# Gradio 및 데이터 처리 라이브러리 임포트 시도
try:
    import gradio as gr
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("="*60)
    print("필수 라이브러리가 설치되지 않았습니다.")
    print("터미널에서 아래 명령어를 실행해주세요:")
    print("pip install gradio pandas matplotlib numpy seaborn")
    print("="*60)
    exit()

# llama.cpp 서버 관련 설정
LLAMA_SERVER_PATH = "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe"
SERVER_HOST = "127.0.0.1"
SERVER_PROCESS = None

# NEO Engine 관련 임포트
try:
    from engine import NEOExecutor
    NEO_ENGINE_AVAILABLE = True
    neo_executor = None
    current_kb_file = None  # 선택된 KB 파일 추적
except ImportError:
    print("⚠️ engine.py를 찾을 수 없습니다. NEO Engine 기능이 비활성화됩니다.")
    NEO_ENGINE_AVAILABLE = False
    neo_executor = None
    current_kb_file = None

# Chroma RAG 임포트
try:
    from chroma_rag import ChromaRAG
    rag_engine = ChromaRAG()
    CHROMA_RAG_AVAILABLE = True
except ImportError:
    print("⚠️ chroma_rag.py를 찾을 수 없습니다. Chroma RAG 기능이 비활성화됩니다.")
    CHROMA_RAG_AVAILABLE = False
    rag_engine = None

# =====================
# 1. 상수/설정
# =====================
SYSTEM_PROMPT_DIR = "./templates"

# 복합질의 단순화 기능 사용 가능 여부 플래그
QUERY_SIMPLIFICATION_AVAILABLE = True

# =====================
# 2. 유틸 함수
# =====================
def safe_dirname(name: str) -> str:
    """Windows에서 사용할 수 없는 문자들을 _로 치환"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def find_gguf_files(dirs: List[str]) -> Dict[str, str]:
    """지정된 디렉토리에서 gguf 모델 파일을 탐색"""
    paths = {}
    for d in dirs:
        if os.path.exists(d):
            for path in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
                name = os.path.splitext(os.path.basename(path))[0]
                paths[name] = path
    print(f"[DEBUG][find_gguf_files] 반환값: {paths}")  # 디버깅 추가
    return paths

# =====================
# 3. 서버/모델 관리 클래스
# =====================
class LlamaServerManager:
    """llama.cpp 서버 실행/중지 및 상태 체크"""
    def __init__(self, server_path: str, host: str = "127.0.0.1"):
        self.server_path = server_path
        self.host = host
        self.process = None
        self.current_model_info = {}

    def start(self, model_path: str, ctx_size: int, port: int, gpu_layers: int) -> bool:
        """서버 실행 및 헬스체크"""
        print(f"[DEBUG][start] model_path 인자: {model_path}")  # 디버깅 추가
        self.stop()
        cmd = [self.server_path, "-m", model_path, "--ctx-size", str(ctx_size), "--host", self.host, "--port", str(port), "-ngl", str(gpu_layers)]
        print(f"[DEBUG] Executing: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            self.current_model_info = {"model_path": model_path, "port": port, "gpu_layers": gpu_layers}
            max_wait, start_time = 60, time.time()
            while time.time() - start_time < max_wait:
                if self.process.poll() is not None:
                    raise ChildProcessError(f"Server process terminated unexpectedly. Stderr: {self.process.stderr.read()}")
                try:
                    response = requests.get(f"http://{self.host}:{port}/v1/models", timeout=2)
                    if response.status_code == 200:
                        print(f"[DEBUG] Server for {model_path} is ready on port {port}.")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(2)
            raise TimeoutError("Server failed to start within the time limit.")
        except Exception as e:
            print(f"[ERROR] Failed to start server: {e}")
            self.stop()
            return False

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                print(f"[DEBUG] Terminating server with PID {self.process.pid}...")
                self.process.terminate()
                self.process.wait(timeout=5)
                print("[DEBUG] Server terminated.")
            except subprocess.TimeoutExpired:
                print("[DEBUG] Server did not terminate in time, killing.")
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None
                self.current_model_info = {}

# =====================
# 4. 모델 벤치마크/평가 클래스
# =====================
class ModelBenchmark:
    """모델별 시스템 프롬프트, 서버, 생성, 평가 관리"""
    def __init__(self, model_paths: Dict[str, str], context_size: int = 2048, system_prompt_dir: str = SYSTEM_PROMPT_DIR):
        self.model_paths = model_paths
        print(f"[DEBUG][ModelBenchmark.__init__] self.model_paths: {self.model_paths}")  # 디버깅 추가
        self.context_size = context_size
        self.system_prompt_dir = Path(system_prompt_dir)
        self.system_prompts = {}
        self.server_manager = LlamaServerManager(LLAMA_SERVER_PATH)
        self.load_system_prompts()

    # --- 시스템 프롬프트 관리 ---
    def load_system_prompts(self):
        """모든 모델의 시스템 프롬프트를 로드"""
        default_prompt_path = self.system_prompt_dir / "default_system_prompt.txt"
        default_prompt = "You are a helpful AI assistant."
        if default_prompt_path.exists():
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                default_prompt = f.read().strip()
        for model_name in self.model_paths:
            self.system_prompts[model_name] = default_prompt
            prefix = model_name.split('-')[0].lower()
            prompt_path = self.system_prompt_dir / f"{prefix}_system_prompt.txt"
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    self.system_prompts[model_name] = f.read().strip()
            specific_prompt_path = self.system_prompt_dir / f"{model_name}_system_prompt.txt"
            if specific_prompt_path.exists():
                with open(specific_prompt_path, 'r', encoding='utf-8') as f:
                    self.system_prompts[model_name] = f.read().strip()

    def set_system_prompt(self, model_name: str, system_prompt: str) -> None:
        self.system_prompts[model_name] = system_prompt

    def get_system_prompt(self, model_name: str) -> str:
        return self.system_prompts.get(model_name, "당신은 유용한 AI 어시스턴트입니다.")

    def load_template_content(self, template_name: str) -> str:
        """템플릿 파일의 내용을 로드합니다."""
        if not template_name:
            return ""
        template_path = self.system_prompt_dir / template_name
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"템플릿 로드 오류: {e}")
                return ""
        return ""

    def save_system_prompt(self, model_name: str, filename: Optional[str] = None) -> None:
        """모델의 시스템 프롬프트를 파일로 저장합니다."""
        if not filename:
            filename = f"{model_name}_system_prompt.txt"
        prompt_path = self.system_prompt_dir / filename
        try:
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(self.system_prompts.get(model_name, ""))
            print(f"시스템 프롬프트가 {prompt_path}에 저장되었습니다.")
        except Exception as e:
            print(f"시스템 프롬프트 저장 오류: {e}")

    # --- 모델 생성/서버 관리 ---
    def generate(self, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float, use_system_prompt: bool, openai_api_key: Optional[str], gemini_api_key: Optional[str], port: int, gpu_layers: int) -> Dict[str, Any]:
        """모델별 생성 API 통합"""
        start_time = time.time()
        system_prompt = self.system_prompts.get(model_name, "") if use_system_prompt else ""
        output = None  # output 변수를 미리 초기화
        try:
            model_type = self.model_paths.get(model_name, "")
            if model_type == "ollama":
                # Ollama API 호출 (삭제 예정)
                pass
            elif model_name.startswith("gpt-"):
                # OpenAI API call logic
                if not openai_api_key: raise ValueError("OpenAI API key is required.")
                headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                messages.append({"role": "user", "content": prompt})
                json_data = {"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=json_data)
                response.raise_for_status()
                data = response.json()
                usage = data.get('usage', {})
                output = data['choices'][0]['message']['content']
                print(f"[DEBUG][{model_name}] 모델 출력 결과 (OpenAI)\n--- SYSTEM PROMPT ---\n{system_prompt}\n--- PROMPT ---\n{prompt}\n--- OUTPUT ---\n{output}\n{'='*60}")
                return {
                    "output": output, "elapsed_time": time.time() - start_time,
                    "tokens_generated": usage.get('completion_tokens', 0),
                    "tokens_prompt": usage.get('prompt_tokens', 0),
                    "tokens_total": usage.get('total_tokens', 0),
                    "system_prompt": system_prompt, "prompt": prompt, "model": model_name
                }

            elif model_name.startswith("gemini"):
                # Gemini API call logic
                if not gemini_api_key: raise ValueError("Gemini API key is required.")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"
                full_prompt = f"{system_prompt}\n\n{prompt}".strip()
                json_data = {"contents": [{"parts": [{"text": full_prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature, "topP": top_p}}
                response = requests.post(url, json=json_data)
                response.raise_for_status()
                data = response.json()
                output = data['candidates'][0]['content']['parts'][0]['text']
                print(f"[DEBUG][{model_name}] 모델 출력 결과 (Gemini)\n--- SYSTEM PROMPT ---\n{system_prompt}\n--- PROMPT ---\n{prompt}\n--- OUTPUT ---\n{output}\n{'='*60}")
                return {
                    "output": output, "elapsed_time": time.time() - start_time, "system_prompt": system_prompt,
                    "prompt": prompt, "model": model_name
                }
                
            else:
                # Local llama.cpp server call logic
                model_path = self.model_paths.get(model_name)
                print(f"[DEBUG] llama.cpp 실행용 model_name: {model_name}, model_path: {model_path}")
                if not model_path:
                    raise ValueError(f"model_paths에 '{model_name}'가 없습니다. 실제 model_paths: {self.model_paths}")
                if not self.server_manager.start(model_path, self.context_size, port, gpu_layers):
                    raise ConnectionError(f"Failed to start local server for {model_name}.")
                headers = {"Content-Type": "application/json"}
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                messages.append({"role": "user", "content": prompt})
                json_data = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
                response = requests.post(f"http://{SERVER_HOST}:{port}/v1/chat/completions", headers=headers, json=json_data)
                response.raise_for_status()
                data = response.json()
                usage = data.get('usage', {})
                output = data['choices'][0]['message']['content']
                print(f"[DEBUG][{model_name}] 모델 출력 결과 (llama.cpp)\n--- SYSTEM PROMPT ---\n{system_prompt}\n--- PROMPT ---\n{prompt}\n--- OUTPUT ---\n{output}\n{'='*60}")
                return {
                    "output": output, "elapsed_time": time.time() - start_time,
                    "tokens_generated": usage.get('completion_tokens', 0),
                    "tokens_prompt": usage.get('prompt_tokens', 0),
                    "tokens_total": usage.get('total_tokens', 0),
                    "system_prompt": system_prompt, "prompt": prompt, "model": model_name
                }
        except Exception as e:
            return {"error": str(e), "prompt": prompt, "model": model_name, "system_prompt": system_prompt, "output": output if output is not None else ""}

    def get_available_kb_files(self) -> List[str]:
        """kb 폴더에서 사용 가능한 모든 .kb 파일 목록을 반환합니다."""
        kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
        kb_files = []
        if os.path.exists(kb_dir):
            for file in os.listdir(kb_dir):
                if file.endswith('.kb'):
                    kb_files.append(file)
        return sorted(kb_files)

    def refresh_kb_files(self):
        """KB 파일 목록을 새로고침합니다."""
        kb_files = self.get_available_kb_files()
        if kb_files:
            return gr.update(choices=kb_files, value=kb_files[0])
        else:
            return gr.update(choices=[], value=None)

    def preprocess_model(self, model_name, port_val, gpu_layers_val, selected_kb_file):
        """모델 서버를 시작하고 연결을 확인합니다."""
        global neo_executor, current_kb_file
        
        if not model_name:
            return "모델을 선택해주세요."
        
        try:
            # NEO Engine 초기화 (가능한 경우)
            neo_status = ""
            if NEO_ENGINE_AVAILABLE and neo_executor is None:
                try:
                    print("NEO Engine 초기화 중...")
                    neo_executor = NEOExecutor()
                    
                    # 선택된 KB 파일 로드
                    kb_loaded = False
                    if selected_kb_file:
                        kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb", selected_kb_file)
                        if os.path.exists(kb_path):
                            if neo_executor.load_kb_file(kb_path):
                                neo_status = f"✅ NEO Engine 초기화 완료 (KB: {selected_kb_file})"
                                kb_loaded = True
                                # 전역 변수에 선택된 KB 파일 저장
                                current_kb_file = selected_kb_file
                            else:
                                neo_status = f"❌ KB 파일 로드 실패: {selected_kb_file}"
                        else:
                            neo_status = f"❌ KB 파일을 찾을 수 없음: {selected_kb_file}"
                    else:
                        # 선택된 파일이 없으면 기본 경로들에서 시도
                        kb_paths = [
                            os.path.join(os.getcwd(), "facts.kb"),
                            os.path.join(os.getcwd(), "NEO", "facts.kb"),
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.kb"),
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "NEO", "facts.kb")
                        ]
                        
                        for kb_path in kb_paths:
                            if os.path.exists(kb_path):
                                if neo_executor.load_kb_file(kb_path):
                                    kb_filename = os.path.basename(kb_path)
                                    neo_status = f"✅ NEO Engine 초기화 완료 (KB: {kb_filename})"
                                    kb_loaded = True
                                    # 전역 변수에 선택된 KB 파일 저장
                                    current_kb_file = kb_filename
                                    break
                        
                        if not kb_loaded:
                            neo_status = "⚠️ NEO Engine 초기화 완료 (KB 파일 없음)"
                            
                except Exception as neo_error:
                    neo_status = f"❌ NEO Engine 초기화 실패: {str(neo_error)}"
                    neo_executor = None
            elif NEO_ENGINE_AVAILABLE and neo_executor is not None:
                neo_status = "✅ NEO Engine 이미 초기화됨"
            else:
                neo_status = "⚠️ NEO Engine 사용 불가"
            
            # 모델 서버 시작
            model_status = ""
            if not model_name.startswith(("gpt-", "gemini")):
                model_path = self.model_paths.get(model_name)
                print(f"로컬 모델 {model_name} 서버 시작 중... (경로: {model_path})")
                if not model_path:
                    model_status = f"❌ {model_name} 경로를 찾을 수 없습니다."
                elif self.server_manager.start(model_path, self.context_size, port_val, gpu_layers_val):
                    model_status = f"✅ {model_name} 서버 시작 완료 (포트: {port_val})"
                else:
                    model_status = f"❌ {model_name} 서버 시작 실패"
            else:
                model_status = f"✅ {model_name} API 모델 준비 완료"
            
            # 전체 상태 반환
            return f"{neo_status}\n{model_status}"
                
        except Exception as e:
            return f"❌ 전처리 중 오류 발생: {str(e)}"

    def get_available_templates(self) -> List[str]:
        """사용 가능한 모든 시스템 프롬프트 템플릿 파일 목록을 반환합니다."""
        templates = []
        if self.system_prompt_dir.exists():
            for file in self.system_prompt_dir.glob("*_system_prompt.txt"):
                templates.append(file.name)
        return sorted(templates)

    def get_available_nkb_files(self) -> List[str]:
        """kb 폴더에서 사용 가능한 모든 .nkb 파일 목록을 반환합니다."""
        kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
        nkb_files = []
        if os.path.exists(kb_dir):
            for file in os.listdir(kb_dir):
                if file.endswith('.nkb'):
                    nkb_files.append(file)
        return sorted(nkb_files)

# =====================
# 5. Gradio UI 생성 함수
# =====================
def create_benchmark_interface(model_paths: Dict[str, str], system_prompt_dir: str):
    """Gradio UI 생성 및 이벤트 핸들러 연결"""
    print(f"[DEBUG][create_benchmark_interface] model_paths: {model_paths}")  # 디버깅 추가
    benchmark = ModelBenchmark(model_paths, system_prompt_dir=system_prompt_dir)
    available_templates = benchmark.get_available_templates()
    available_nkb_files = benchmark.get_available_nkb_files()
    
    with gr.Blocks(title="sLLM 벤치마크", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# sLLM 모델 벤치마크\n**주의**: 이 도구는 단일 사용자용입니다. 여러 사용자가 동시에 로컬 모델(GGUF)을 테스트하면 서버 충돌이 발생할 수 있습니다.")
        
        openai_key = gr.Textbox(visible=False)
        gemini_key = gr.Textbox(visible=False)

        with gr.Tab("벤치마크 실행 및 시각화"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Settings Column
                    gr.Markdown("### 1. 벤치마크 설정")
                    benchmark_models = gr.CheckboxGroup(choices=list(model_paths.keys()), label="모델 선택", value=list(model_paths.keys()))
                    query_file = gr.File(label="쿼리 JSON 파일", file_types=[".json"])
                    category_checkboxes = gr.CheckboxGroup(label="카테고리 선택")
                    with gr.Accordion("상세 파라미터", open=False):
                        use_sys_prompt = gr.Checkbox(value=True, label="시스템 프롬프트 사용")
                        # 시스템 프롬프트 템플릿 드롭다운 (동적 표시)
                        sys_prompt_template_dropdown = gr.Dropdown(
                            choices=[""] + [f for f in os.listdir(system_prompt_dir) if f.endswith('.txt')],
                            label="시스템 프롬프트 템플릿 선택",
                            visible=use_sys_prompt.value,
                            allow_custom_value=False
                        )
                        sys_prompt_template_content = gr.Textbox(lines=4, label="시스템 프롬프트 내용", visible=use_sys_prompt.value)
                        use_query_simplification = gr.Checkbox(value=False, label="복합질의 단순화 사용", interactive=QUERY_SIMPLIFICATION_AVAILABLE)
                        if not QUERY_SIMPLIFICATION_AVAILABLE:
                            gr.Markdown("⚠️ **복합질의 단순화**: ollama_preprocessor.py를 찾을 수 없어 비활성화되었습니다.")
                        use_output_evaluation = gr.Checkbox(value=False, label="출력 평가 및 재생성 사용")
                        evaluation_model = gr.Dropdown(choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] + [m for m in model_paths.keys() if not m.startswith(("gpt-", "gemini"))], 
                                                      value="gpt-4o", label="평가용 모델 선택", visible=False)
                        gr.Markdown("**주의**: 로컬 모델을 평가 모델로 사용할 경우 서버 부하가 증가할 수 있습니다. OpenAI 모델 사용을 권장합니다.", visible=False)
                        max_retries = gr.Slider(1, 5, value=3, step=1, label="최대 재시도 횟수", visible=False)
                        temp = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                        top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-p")
                        max_tokens = gr.Slider(16, 4096, value=512, step=16, label="최대 토큰")
                    with gr.Accordion("로컬 서버 설정 (GGUF)", open=False):
                        gpu_layers = gr.Slider(0, 128, value=100, step=1, label="GPU Layers (-ngl)")
                        port = gr.Number(value=8080, label="서버 포트")
                    prompts_textbox = gr.Textbox(lines=5, label="프롬프트 (JSON 미사용시)")
                    run_button = gr.Button("벤치마크 실행", variant="primary")

                with gr.Column(scale=2):
                    # Results Column
                    gr.Markdown("### 2. 진행 상황 및 결과")
                    progress_output = gr.Textbox(label="진행 상황", interactive=False)
                    results_json = gr.JSON(label="결과 JSON")
                    gr.Markdown("### 3. 결과 저장 및 시각화")
                    save_button = gr.Button("결과 저장 및 시각화")
                    save_status = gr.Textbox(label="저장 상태", interactive=False)
                    # 수정된 코드
                    with gr.Column(variant="panel"): # 'panel' variant가 Box와 유사한 스타일을 제공합니다.
                        gr.Markdown("#### 시각화 옵션")
                        with gr.Row():
                            viz_type = gr.Radio(
                                choices=["성공률", "평균 처리 시간", "평균 토큰/초"],
                                value="성공률",
                                label="시각화 메트릭"
                            )

                        with gr.Row():
                            viz_type = gr.Radio(
                                choices=["성공률", "평균 처리 시간", "평균 토큰/초", "평균 평가 점수", "평균 재시도 횟수"],
                                value="성공률",
                                label="시각화 메트릭"
                            )
                            summary_file = gr.File(label="요약 CSV 파일", interactive=False)
                    plot_output = gr.Plot(label="시각화 결과")
                    summary_df_output = gr.Dataframe(label="성능 요약")

        with gr.Tab("질의 생성 벤치마크"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 질의 생성 벤치마크 설정")
                    querygen_models = gr.CheckboxGroup(choices=list(model_paths.keys()), label="모델 선택", value=list(model_paths.keys()))
                    querygen_query_file = gr.File(label="쿼리 JSON 파일", file_types=[".json"])
                    querygen_category_checkboxes = gr.CheckboxGroup(label="카테고리 선택")
                    querygen_rag_file = gr.Dropdown(choices=available_nkb_files, label="RAG 파일(.nkb) 선택", allow_custom_value=False)
                    with gr.Accordion("상세 파라미터", open=False):
                        querygen_use_sys_prompt = gr.Checkbox(value=True, label="시스템 프롬프트 사용")
                        querygen_use_query_simplification = gr.Checkbox(value=False, label="복합질의 단순화 사용", interactive=QUERY_SIMPLIFICATION_AVAILABLE)
                        if not QUERY_SIMPLIFICATION_AVAILABLE:
                            gr.Markdown("⚠️ **복합질의 단순화**: ollama_preprocessor.py를 찾을 수 없어 비활성화되었습니다.")
                        querygen_temp = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                        querygen_top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-p")
                        querygen_max_tokens = gr.Slider(16, 4096, value=512, step=16, label="최대 토큰")
                    querygen_prompts_textbox = gr.Textbox(lines=5, label="프롬프트 (JSON 미사용시)")
                    querygen_run_button = gr.Button("질의 생성 실행", variant="primary")
                    # 시각화 옵션 및 결과 저장
                    querygen_save_button = gr.Button("결과 저장 및 시각화")
                    querygen_save_status = gr.Textbox(label="저장 상태", interactive=False)
                    with gr.Column(variant="panel"):
                        gr.Markdown("#### 시각화 옵션")
                        with gr.Row():
                            querygen_viz_type = gr.Radio(
                                choices=["성공률", "평균 처리 시간", "평균 토큰/초"],
                                value="성공률",
                                label="시각화 메트릭"
                            )
                        with gr.Row():
                            querygen_viz_type = gr.Radio(
                                choices=["성공률", "평균 처리 시간", "평균 토큰/초", "평균 평가 점수", "평균 재시도 횟수"],
                                value="성공률",
                                label="시각화 메트릭"
                            )
                            querygen_summary_file = gr.File(label="요약 CSV 파일", interactive=False)
                with gr.Column(scale=2):
                    gr.Markdown("### 2. 진행 상황 및 결과")
                    querygen_progress_output = gr.Textbox(label="진행 상황", interactive=False)
                    querygen_results_json = gr.JSON(label="결과 JSON")
                    querygen_plot_output = gr.Plot(label="시각화 결과")
                    querygen_summary_df_output = gr.Dataframe(label="성능 요약")
            
            # 핸들러: 쿼리 파일/카테고리 연동
            def querygen_load_queries(file):
                if not file: return gr.update(choices=[], value=[]), ""
                with open(file.name, 'r', encoding='utf-8') as f: data = json.load(f)
                categories = list(data.keys())
                all_prompts = [p for cat_prompts in data.values() for p in cat_prompts]
                return gr.update(choices=categories, value=categories), "\n".join(all_prompts)
            querygen_query_file.change(querygen_load_queries, querygen_query_file, [querygen_category_checkboxes, querygen_prompts_textbox])
            def querygen_filter_prompts(file, selected_cats):
                if not file or not selected_cats: return ""
                with open(file.name, 'r', encoding='utf-8') as f: data = json.load(f)
                prompts = [p for cat, cat_prompts in data.items() if cat in selected_cats for p in cat_prompts]
                return "\n".join(prompts)
            querygen_category_checkboxes.change(querygen_filter_prompts, [querygen_query_file, querygen_category_checkboxes], querygen_prompts_textbox)
            # 실행 핸들러
            def run_querygen_benchmark_task(models, use_sys, use_simplification, q_file, sel_cats, rag_file, prompts_str, temp_val, top_p_val, max_tok_val, progress=gr.Progress(track_tqdm=True)):
                prompts = [p.strip() for p in prompts_str.split('\n') if p.strip()]
                if not prompts: raise gr.Error("No prompts specified.")
                # 복합질의 단순화 적용
                if use_simplification and QUERY_SIMPLIFICATION_AVAILABLE:
                    progress(0, desc="복합질의 단순화 처리 중...")
                    simplified_prompts = []
                    original_to_simplified = {}
                    for i, prompt in enumerate(prompts):
                        progress(i / len(prompts), desc=f"질의 단순화 중... ({i+1}/{len(prompts)})")
                        simplified = simplify_sentence(prompt)
                        simplified_prompts.extend(simplified)
                        original_to_simplified[prompt] = simplified
                    prompts = simplified_prompts
                    progress(1, desc="단순화 완료")
                cat_map = {}
                if q_file:
                    with open(q_file.name, 'r', encoding='utf-8') as f: data = json.load(f)
                    for cat, cat_prompts in data.items():
                        if cat in sel_cats:
                            for p in cat_prompts:
                                if use_simplification and QUERY_SIMPLIFICATION_AVAILABLE:
                                    if p.strip() in original_to_simplified:
                                        for simplified_p in original_to_simplified[p.strip()]:
                                            cat_map[simplified_p] = cat
                                else:
                                    cat_map[p.strip()] = cat
                # RAG 파일(.nkb) 내용 읽기
                rag_context = ""
                if rag_file:
                    nkb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb", rag_file)
                    if os.path.exists(nkb_path):
                        with open(nkb_path, 'r', encoding='utf-8') as f:
                            rag_context = f.read().strip()
                total_tasks = len(models) * len(prompts)
                results_data = {"detailed": {m: [] for m in models}}
                for model_idx, model in enumerate(models):
                    for prompt_idx, prompt in enumerate(prompts):
                        task_num = model_idx * len(prompts) + prompt_idx
                        progress(task_num / total_tasks, desc=f"({task_num+1}/{total_tasks}) {model}")
                        # system prompt에 RAG context prepend
                        system_prompt = ""
                        if use_sys:
                            system_prompt = benchmark.get_system_prompt(model)
                        if rag_context:
                            system_prompt = f"[RAG 문서]\n{rag_context}\n\n" + system_prompt
                        result = benchmark.generate(model, prompt, max_tok_val, temp_val, top_p_val, False, None, None, 8080, 100)
                        # system_prompt를 강제로 덮어씌움
                        result['system_prompt'] = system_prompt
                        result['category'] = cat_map.get(prompt, 'N/A')
                        results_data["detailed"][model].append(result)
                benchmark.server_manager.stop()
                return f"질의 생성 벤치마크 완료: 총 {total_tasks}개 태스크 실행", results_data
            querygen_run_button.click(
                run_querygen_benchmark_task,
                [querygen_models, querygen_use_sys_prompt, querygen_use_query_simplification, querygen_query_file, querygen_category_checkboxes, querygen_rag_file, querygen_prompts_textbox, querygen_temp, querygen_top_p, querygen_max_tokens],
                [querygen_progress_output, querygen_results_json]
            )

        with gr.Tab("단일 모델 테스트 (RAG)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1. 지식(.nkb) 생성")
                    single_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="모델 선택")
                    single_system_prompt_template = gr.Dropdown(choices=[""] + available_templates, label="시스템 프롬프트 템플릿 로드", allow_custom_value=False)
                    single_system_prompt_input = gr.Textbox(lines=4, label="시스템 프롬프트", placeholder="시스템 프롬프트를 입력하거나 템플릿을 선택하세요...")
                    single_prompt_input = gr.Textbox(lines=5, label="프롬프트", placeholder="지식(.nkb)로 저장할 프롬프트를 입력하세요...")
                    single_use_simplification = gr.Checkbox(value=False, label="복합질의 단순화 사용", interactive=QUERY_SIMPLIFICATION_AVAILABLE)
                    with gr.Row():
                        single_temp = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                        single_top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-p")
                    single_max_tokens = gr.Slider(16, 4096, value=512, step=16, label="최대 토큰")
                    single_run_button = gr.Button("지식 생성 및 저장")
                    single_output_text = gr.Textbox(lines=20, label="모델 출력", max_lines=40)
                with gr.Column():
                    gr.Markdown("#### 2. RAG(검색 기반 생성)")
                    rag_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="RAG용 모델 선택")
                    rag_system_prompt_template = gr.Dropdown(choices=[""] + available_templates, label="RAG 시스템 프롬프트 템플릿 로드", allow_custom_value=False)
                    rag_system_prompt_input = gr.Textbox(lines=4, label="RAG용 시스템 프롬프트", placeholder="RAG용 시스템 프롬프트를 입력하거나 템플릿을 선택하세요...")
                    rag_nkb_dropdown = gr.Dropdown(choices=available_nkb_files, label="지식 파일(.nkb) 선택", allow_custom_value=False)
                    rag_prompt_input = gr.Textbox(lines=3, label="RAG 프롬프트", placeholder="선택한 .nkb 파일을 참고하여 답변할 프롬프트를 입력하세요.")
                    rag_run_button = gr.Button("nkb 기반 응답 실행")
                    rag_output_text = gr.Textbox(lines=20, label="RAG 출력 결과", max_lines=40)

        with gr.Tab("시스템 프롬프트 관리"):
            with gr.Row():
                with gr.Column():
                    manage_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="모델 선택")
                    manage_template_dropdown = gr.Dropdown(choices=[""] + available_templates, label="템플릿 선택", allow_custom_value=False)
                    load_template_button = gr.Button("템플릿 로드")
                    manage_system_prompt = gr.Textbox(lines=8, label="시스템 프롬프트", placeholder="시스템 프롬프트를 입력하세요...")
                    with gr.Row():
                        load_prompt_button = gr.Button("모델 프롬프트 로드")
                        save_prompt_button = gr.Button("저장")
                    template_name = gr.Textbox(label="템플릿 파일명 (저장 시 사용)", placeholder="예: custom_system_prompt.txt")
                with gr.Column():
                    system_prompt_status = gr.Textbox(label="상태")
                    available_templates_text = gr.Textbox(label="사용 가능한 템플릿", value="\n".join(available_templates) or "사용 가능한 템플릿이 없습니다.", interactive=False)

        with gr.Tab("API 키 관리"):
            gr.Markdown("### API 키 설정\n벤치마크 실행 전 API 키를 입력하고 저장해주세요.")
            openai_input = gr.Textbox(label="OpenAI API Key", type="password")
            gemini_input = gr.Textbox(label="Gemini API Key", type="password")
            save_keys_btn = gr.Button("API 키 저장")
            save_keys_btn.click(lambda o, g: (o, g), [openai_input, gemini_input], [openai_key, gemini_key]).then(lambda: gr.Info("API Keys saved for session."))

        with gr.Tab("QA 챗봇"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 챗봇 설정")
                    chatbot_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="챗봇 모델 선택", value=list(model_paths.keys())[0] if model_paths else None)
                    chatbot_system_prompt_template = gr.Dropdown(choices=[""] + available_templates, label="시스템 프롬프트 템플릿 로드", allow_custom_value=False)
                    chatbot_system_prompt_input = gr.Textbox(lines=4, label="시스템 프롬프트", placeholder="챗봇의 성격과 역할을 정의하세요...", value="당신은 유용하고 친근한 AI 어시스턴트입니다. 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.")
                    
                    with gr.Accordion("생성 파라미터", open=False):
                        chatbot_temp = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                        chatbot_top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-p")
                        chatbot_max_tokens = gr.Slider(16, 4096, value=512, step=16, label="최대 토큰")
                    
                    with gr.Accordion("고급 옵션", open=False):
                        chatbot_use_evaluation = gr.Checkbox(value=False, label="출력 평가 및 재생성 사용")
                        chatbot_evaluation_model = gr.Dropdown(choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] + [m for m in model_paths.keys() if not m.startswith(("gpt-", "gemini"))], 
                                                              value="gpt-4o", label="평가용 모델 선택", visible=False)
                        chatbot_max_retries = gr.Slider(1, 5, value=3, step=1, label="최대 재시도 횟수", visible=False)
                    
                    # Agent 설정 섹션 추가
                    gr.Markdown("### Agent 설정")
                    use_agent_dialogue = gr.Checkbox(value=True, label="Agent 대화 모드 사용", info="사용자와 직접 대화하여 정보를 구조화한 후 NEO Query로 변환")
                    agent_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="Agent 모델 선택", value=list(model_paths.keys())[0] if model_paths else None)
                    agent_system_prompt_template = gr.Dropdown(choices=["default_system_prompt.txt"] + available_templates, label="Agent 시스템 프롬프트 템플릿", value="default_system_prompt.txt", allow_custom_value=False)
                    agent_system_prompt_input = gr.Textbox(lines=6, label="Agent 시스템 프롬프트", placeholder="Agent의 역할과 지침을 정의하세요...", value=benchmark.load_template_content("default_system_prompt.txt") if os.path.exists(os.path.join(system_prompt_dir, "default_system_prompt.txt")) else "", interactive=False)
                    
                    with gr.Accordion("Agent 생성 파라미터", open=False):
                        agent_temp = gr.Slider(0.0, 2.0, value=0.3, label="Agent Temperature", info="낮은 값으로 일관된 응답")
                        agent_top_p = gr.Slider(0.0, 1.0, value=0.9, label="Agent Top-p")
                        agent_max_tokens = gr.Slider(16, 2048, value=256, step=16, label="Agent 최대 토큰")
                    
                    gr.Markdown("### 전처리")
                    preprocess_btn = gr.Button("모델 서버 시작 및 연결 확인", variant="primary")
                    preprocess_status = gr.Textbox(label="전처리 상태", interactive=False, value="전처리가 필요합니다.")
                    
                    # KB 파일 선택 추가
                    gr.Markdown("### KB 파일 선택")
                    kb_file_dropdown = gr.Dropdown(choices=benchmark.get_available_kb_files(), label="KB 파일 선택", allow_custom_value=False, value=benchmark.get_available_kb_files()[0] if benchmark.get_available_kb_files() else None)
                    refresh_kb_btn = gr.Button("KB 파일 목록 새로고침", variant="secondary", size="sm")
                    
                    with gr.Row():
                        clear_chat_btn = gr.Button("대화 기록 지우기", variant="secondary")
                        save_chat_btn = gr.Button("대화 저장", variant="secondary")
                        load_chat_btn = gr.Button("대화 불러오기", variant="secondary")
                    
                    chat_file = gr.File(label="대화 파일 (.json)", file_types=[".json"], visible=False)
                    
                with gr.Column(scale=2):
                    gr.Markdown("### 대화")
                    chatbot = gr.Chatbot(label="챗봇", height=800)
                    chatbot_input = gr.Textbox(lines=2, label="메시지 입력", placeholder="메시지를 입력하세요...")
                    
                    with gr.Row():
                        send_btn = gr.Button("전송", variant="primary")
                        stop_btn = gr.Button("중지", variant="stop")
                    
                    chatbot_status = gr.Textbox(label="상태", interactive=False, value="챗봇이 준비되었습니다.")

        with gr.Tab("지식베이스 생성"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 지식베이스 생성 설정")
                    kb_gen_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="생성 모델 선택", value=list(model_paths.keys())[0] if model_paths else None)
                    kb_gen_system_prompt_template = gr.Dropdown(choices=[""] + available_templates, label="시스템 프롬프트 템플릿 로드", allow_custom_value=False)
                    kb_gen_system_prompt_input = gr.Textbox(lines=6, label="시스템 프롬프트", placeholder="지식베이스 생성을 위한 시스템 프롬프트를 입력하세요...", value="당신은 전문적인 지식베이스 생성 전문가입니다. 주어진 문서나 텍스트에서 모든 중요한 지식 항목을 추출하여 구조화된 지식베이스를 생성해주세요. 문서에 포함된 모든 관련 정보를 빠짐없이 추출하여 여러 개의 지식 항목으로 변환하세요.", interactive=True)
                    
                    with gr.Accordion("생성 파라미터", open=False):
                        kb_gen_temp = gr.Slider(0.0, 2.0, value=0.3, label="Temperature", info="낮은 값으로 일관된 생성")
                        kb_gen_top_p = gr.Slider(0.0, 1.0, value=0.9, label="Top-p")
                        kb_gen_max_tokens = gr.Slider(16, 4096, value=1024, step=16, label="최대 토큰")
                    
                    gr.Markdown("### 입력 데이터")
                    kb_gen_input_type = gr.Radio(
                        choices=["텍스트 입력", "파일 업로드", "URL 입력"],
                        value="텍스트 입력",
                        label="입력 방식 선택"
                    )
                    
                    kb_gen_text_input = gr.Textbox(
                        lines=10, 
                        label="텍스트 입력", 
                        placeholder="지식베이스로 변환할 텍스트를 입력하세요...",
                        visible=True
                    )
                    
                    kb_gen_file_input = gr.File(
                        label="파일 업로드 (.txt, .md, .pdf, .docx)",
                        file_types=[".txt", ".md", ".pdf", ".docx"],
                        visible=False
                    )
                    
                    kb_gen_url_input = gr.Textbox(
                        lines=2,
                        label="URL 입력",
                        placeholder="https://example.com/article",
                        visible=False
                    )
                    
                    gr.Markdown("### 생성 옵션")
                    kb_gen_output_format = gr.Radio(
                        choices=["NEO 형식 (.kb)", "자연어 형식 (.nkb)", "JSON 형식 (.json)"],
                        value="NEO 형식 (.kb)",
                        label="출력 형식"
                    )
                    
                    kb_gen_filename = gr.Textbox(
                        label="파일명 (확장자 제외)",
                        placeholder="예: health_insurance_kb",
                        value="generated_kb"
                    )
                    
                    kb_gen_button = gr.Button("지식베이스 생성", variant="primary")
                    
                with gr.Column(scale=2):
                    gr.Markdown("### 생성 결과")
                    kb_gen_progress = gr.Textbox(label="진행 상황", interactive=False, value="지식베이스 생성을 기다리는 중...")
                    kb_gen_output = gr.Textbox(
                        lines=20, 
                        label="생성된 지식베이스", 
                        placeholder="생성된 지식베이스가 여기에 표시됩니다...",
                        max_lines=50
                    )
                    
                    with gr.Row():
                        kb_gen_save_btn = gr.Button("파일 저장", variant="secondary")
                        kb_gen_clear_btn = gr.Button("결과 지우기", variant="secondary")
                        kb_gen_load_btn = gr.Button("생성된 KB 로드", variant="secondary")
                    
                    kb_gen_save_status = gr.Textbox(label="저장 상태", interactive=False)
                    
                    gr.Markdown("### 생성된 KB 파일 목록")
                    kb_gen_file_list = gr.Dropdown(
                        choices=[],
                        label="생성된 KB 파일 선택",
                        allow_custom_value=False
                    )
                    refresh_kb_gen_btn = gr.Button("파일 목록 새로고침", variant="secondary", size="sm")

        with gr.Tab("단일 모델 테스트 (Native)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Native 단일 모델 테스트 (대화형)")
                    native_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="모델 선택")
                    native_system_prompt_template = gr.Dropdown(choices=[""] + available_templates, label="시스템 프롬프트 템플릿 로드", allow_custom_value=False)
                    native_system_prompt_input = gr.Textbox(lines=4, label="시스템 프롬프트", placeholder="시스템 프롬프트를 입력하거나 템플릿을 선택하세요...")
                    native_chatbot = gr.Chatbot(label="대화", height=800)
                    native_prompt_input = gr.Textbox(lines=2, label="메시지 입력", placeholder="메시지를 입력하세요...")
                    with gr.Row():
                        native_send_btn = gr.Button("전송", variant="primary")
                        native_clear_btn = gr.Button("대화 기록 지우기", variant="secondary")
                    native_status = gr.Textbox(label="상태", interactive=False, value="Native 대화형 테스트 준비됨.")
                    with gr.Row():
                        native_temp = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                        native_top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-p")
                    native_max_tokens = gr.Slider(16, 4096, value=512, step=16, label="최대 토큰")

        # Event Handlers
        def load_queries(file):
            if not file: return gr.update(choices=[], value=[]), ""
            with open(file.name, 'r', encoding='utf-8') as f: data = json.load(f)
            categories = list(data.keys())
            all_prompts = [p for cat_prompts in data.values() for p in cat_prompts]
            return gr.update(choices=categories, value=categories), "\n".join(all_prompts)
        query_file.change(load_queries, query_file, [category_checkboxes, prompts_textbox])

        def filter_prompts(file, selected_cats):
            if not file or not selected_cats: return ""
            with open(file.name, 'r', encoding='utf-8') as f: data = json.load(f)
            prompts = [p for cat, cat_prompts in data.items() if cat in selected_cats for p in cat_prompts]
            return "\n".join(prompts)
        category_checkboxes.change(filter_prompts, [query_file, category_checkboxes], prompts_textbox)

        def toggle_evaluation_ui(use_evaluation):
            """출력 평가 옵션에 따라 관련 UI 요소들의 가시성을 조절합니다."""
            return gr.update(visible=use_evaluation), gr.update(visible=use_evaluation), gr.update(visible=use_evaluation)

        def check_evaluation_model(model_name):
            """평가 모델 선택 시 주의사항을 표시합니다."""
            if model_name and not model_name.startswith(("gpt-", "gemini")):
                return "⚠️ **주의**: 로컬 모델을 평가 모델로 사용할 경우 서버 부하가 증가하고 타임아웃이 발생할 수 있습니다. OpenAI 모델 사용을 강력히 권장합니다."
            return ""

        use_output_evaluation.change(
            toggle_evaluation_ui,
            use_output_evaluation,
            [evaluation_model, max_retries, gr.Markdown("**주의**: 로컬 모델을 평가 모델로 사용할 경우 서버 부하가 증가할 수 있습니다. OpenAI 모델 사용을 권장합니다.")]
        )

        evaluation_model.change(
            check_evaluation_model,
            evaluation_model,
            gr.Markdown("")
        )

        def run_benchmark_task(models, use_sys, use_simplification, use_evaluation, eval_model, max_retry, q_file, sel_cats, prompts_str, temp_val, top_p_val, max_tok_val, gpu_l, port_val, oai_key, gem_key, progress=gr.Progress(track_tqdm=True), sys_prompt_template_content=None):
            # 시스템 프롬프트 템플릿이 선택되어 있으면 각 모델에 실제로 반영
            if use_sys and sys_prompt_template_content:
                for model in models:
                    benchmark.set_system_prompt(model, sys_prompt_template_content)
            prompts = [p.strip() for p in prompts_str.split('\n') if p.strip()]
            if not prompts: raise gr.Error("No prompts specified.")

            # 복합질의 단순화 적용
            if use_simplification and QUERY_SIMPLIFICATION_AVAILABLE:
                progress(0, desc="복합질의 단순화 처리 중...")
                simplified_prompts = []
                original_to_simplified = {}  # 원본 프롬프트와 단순화된 프롬프트들의 매핑
                
                for i, prompt in enumerate(prompts):
                    progress(i / len(prompts), desc=f"질의 단순화 중... ({i+1}/{len(prompts)})")
                    simplified = simplify_sentence(prompt)
                    simplified_prompts.extend(simplified)
                    original_to_simplified[prompt] = simplified
                
                prompts = simplified_prompts
                progress(1, desc="단순화 완료")

            cat_map = {}
            if q_file:
                with open(q_file.name, 'r', encoding='utf-8') as f: data = json.load(f)
                for cat, cat_prompts in data.items():
                    if cat in sel_cats:
                        for p in cat_prompts: 
                            if use_simplification and QUERY_SIMPLIFICATION_AVAILABLE:
                                # 단순화된 프롬프트들에 대해 카테고리 매핑
                                if p.strip() in original_to_simplified:
                                    for simplified_p in original_to_simplified[p.strip()]:
                                        cat_map[simplified_p] = cat
                                else:
                                    cat_map[p.strip()] = cat
            
            total_tasks = len(models) * len(prompts)
            results_data = {"detailed": {m: [] for m in models}}

            for model_idx, model in enumerate(models):
                for prompt_idx, prompt in enumerate(prompts):
                    task_num = model_idx * len(prompts) + prompt_idx
                    progress(task_num / total_tasks, desc=f"({task_num+1}/{total_tasks}) {model}")
                    
                    # 출력 평가 및 재생성 기능 사용 여부에 따라 다른 생성 방식 사용
                    if use_evaluation:
                        progress(task_num / total_tasks, desc=f"({task_num+1}/{total_tasks}) {model} - 평가 모드")
                        result = generate_with_retry(benchmark, model, prompt, max_tok_val, temp_val, top_p_val, 
                                                   use_sys, oai_key, gem_key, port_val, gpu_l, eval_model, max_retry, system_prompt_dir)
                    else:
                        progress(task_num / total_tasks, desc=f"({task_num+1}/{total_tasks}) {model}")
                        result = benchmark.generate(model, prompt, max_tok_val, temp_val, top_p_val, use_sys, oai_key, gem_key, port_val, gpu_l)
                    
                    result['category'] = cat_map.get(prompt, 'N/A')
                    results_data["detailed"][model].append(result)
                    
                    # API 모델의 경우 요청 속도 제한을 피하기 위해 딜레이 추가
                    if model.startswith("gpt-") or model.startswith("gemini"):
                        time.sleep(1) # 1초 지연

            benchmark.server_manager.stop()
            return f"벤치마크 완료: 총 {total_tasks}개 태스크 실행", results_data

        run_button.click(
            run_benchmark_task, 
            [
                benchmark_models, use_sys_prompt, use_query_simplification, use_output_evaluation, evaluation_model, max_retries,
                query_file, category_checkboxes, prompts_textbox, temp, top_p, max_tokens, gpu_layers, port, openai_key, gemini_key, sys_prompt_template_content
            ], 
            [progress_output, results_json]
        )

        def save_and_plot(results, viz_choice):
            if not results or "detailed" not in results:
                raise gr.Error("저장할 결과가 없습니다.")
            
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join("benchmark_results", timestamp)
            os.makedirs(output_dir, exist_ok=True)

            # --- Detailed Results Saving ---
            all_data = []
            for model, model_results in results["detailed"].items():
                model_dir = os.path.join(output_dir, safe_dirname(model))
                os.makedirs(model_dir, exist_ok=True)
                
                # Save detailed JSON for each model
                model_json_path = os.path.join(model_dir, f"{safe_dirname(model)}_detailed_{timestamp}.json")
                with open(model_json_path, 'w', encoding='utf-8') as f:
                    json.dump(model_results, f, ensure_ascii=False, indent=2)

                # Prepare data for CSV
                for res in model_results:
                    row = res.copy()
                    row['is_error'] = 1 if 'error' in res else 0
                    
                    # 평가 정보 추가
                    if 'evaluation_results' in res:
                        row['attempts'] = res.get('attempts', 1)
                        row['final_score'] = res['evaluation_results'][-1]['score'] if res['evaluation_results'] else 0
                        row['evaluation_model'] = res['evaluation_results'][-1]['evaluation_model'] if res['evaluation_results'] else 'N/A'
                        row['evaluation_feedback'] = res['evaluation_results'][-1]['feedback'] if res['evaluation_results'] else 'N/A'
                    else:
                        row['attempts'] = 1
                        row['final_score'] = 'N/A'
                        row['evaluation_model'] = 'N/A'
                        row['evaluation_feedback'] = 'N/A'
                    
                    all_data.append(row)
            
            if not all_data:
                raise gr.Error("결과 데이터가 비어있습니다.")

            df = pd.DataFrame(all_data)
            
            # Save detailed CSV for each model
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                model_csv_path = os.path.join(output_dir, safe_dirname(model), f"{safe_dirname(model)}_detailed_{timestamp}.csv")
                model_df.to_csv(model_csv_path, index=False, encoding='utf-8')


            # --- Summary Calculation and Saving ---
            success_df = df[df['is_error'] == 0].copy()
            
            summary = df.groupby(['model', 'category'])['is_error'].count().reset_index(name='total')
            success_counts = success_df.groupby(['model', 'category']).size().reset_index(name='success')
            summary = pd.merge(summary, success_counts, on=['model', 'category'], how='left').fillna(0)
            summary['success'] = summary['success'].astype(int)
            summary['fail'] = summary['total'] - summary['success']
            summary['성공률'] = (summary['success'] / summary['total'] * 100).round(2)
            
            if not success_df.empty:
                success_df.loc[:, 'tps'] = success_df['tokens_generated'] / success_df['elapsed_time'].replace(0, 1e-9)
                
                # 기본 성능 메트릭
                perf_summary = success_df.groupby(['model', 'category']).agg(
                    평균_처리_시간=('elapsed_time', 'mean'),
                    평균_토큰_초=('tps', 'mean'),
                    평균_생성_토큰=('tokens_generated', 'mean')
                ).reset_index()
                summary = pd.merge(summary, perf_summary, on=['model', 'category'], how='left').fillna(0)
                
                # 평가 관련 메트릭 (평가가 수행된 경우에만)
                if 'final_score' in success_df.columns and success_df['final_score'].dtype != 'object':
                    eval_summary = success_df.groupby(['model', 'category']).agg(
                        평균_평가_점수=('final_score', 'mean'),
                        평균_재시도_횟수=('attempts', 'mean')
                    ).reset_index()
                    summary = pd.merge(summary, eval_summary, on=['model', 'category'], how='left').fillna(0)

            summary_path = os.path.join(output_dir, f"summary_{timestamp}.csv")
            summary.to_csv(summary_path, index=False, encoding='utf-8')
            
            # --- Plotting ---
            fig, table = plot_summary(summary, viz_choice)
            
            return f"결과 저장 완료: {output_dir}", summary_path, fig, table
        
        def plot_summary(df, choice):
            if df.empty: return None, pd.DataFrame()
            
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 7))
            
            metric_map = {
                "성공률": "성공률", 
                "평균 처리 시간": "평균_처리_시간", 
                "평균 토큰/초": "평균_토큰_초",
                "평균 생성 토큰": "평균_생성_토큰",
                "평균 평가 점수": "평균_평가_점수",
                "평균 재시도 횟수": "평균_재시도_횟수",
                "성공/실패 건수": "success" # Special case
            }
            metric = metric_map.get(choice)

            if metric is None: return None, df
            
            if choice == "성공/실패 건수":
                df_pivot = df.pivot(index='category', columns='model', values=['success', 'fail'])
                df_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
                ax.set_ylabel("프롬프트 수")
            else:
                if metric not in df.columns: return None, df
                df.pivot(index='category', columns='model', values=metric).plot(kind='bar', ax=ax)
                ax.set_ylabel(choice)

            ax.set_title(f'모델 및 카테고리별 {choice} 비교', fontsize=16)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Model')
            fig.tight_layout()
            
            # Return the summary dataframe for display
            return fig, df.round(2)

        save_button.click(
            save_and_plot, 
            [results_json, viz_type], 
            [save_status, summary_file, plot_output, summary_df_output]
        )
        
        viz_type.change(
            lambda file, choice: plot_summary(pd.read_csv(file.name), choice) if file else (None, pd.DataFrame()),
            [summary_file, viz_type],
            [plot_output, summary_df_output]
        )
        
        # --- Handlers for New Tabs ---
        def update_nkb_dropdown():
            """kb 폴더의 nkb 파일 목록을 업데이트합니다."""
            nkb_files = benchmark.get_available_nkb_files()
            return gr.update(choices=nkb_files, value=nkb_files[0] if nkb_files else None)

        def run_single_model_task(model, sys_prompt, prompt, use_simplification, temp, top_p, max_tok, gpu_l, port_val, oai_key, gem_key):
            try:
                # 복합질의 단순화 적용
                if use_simplification and QUERY_SIMPLIFICATION_AVAILABLE:
                    simplified_prompts = simplify_sentence(prompt)
                    if len(simplified_prompts) > 1:
                        # 여러 개의 단순화된 프롬프트가 있는 경우, 첫 번째 것만 사용
                        prompt = simplified_prompts[0]
                        print(f"복합질의 단순화 적용: {prompt}")
                
                benchmark.set_system_prompt(model, sys_prompt)
                result = benchmark.generate(model, prompt, max_tok, temp, top_p, True, oai_key, gem_key, port_val, gpu_l)
                if 'error' in result:
                    return f"오류: {result['error']}"
                
                # 지식 파일 저장
                output = result['output']
                kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
                os.makedirs(kb_dir, exist_ok=True)
                nkb_path = os.path.join(kb_dir, "test.nkb")
                with open(nkb_path, 'w', encoding='utf-8') as f:
                    f.write(str(output))

                return output
            except Exception as e:
                return f"실행 중 예외 발생: {e}"

        single_run_button.click(
            run_single_model_task,
            [single_model_dropdown, single_system_prompt_input, single_prompt_input, single_use_simplification, single_temp, single_top_p, single_max_tokens, gpu_layers, port, openai_key, gemini_key],
            single_output_text
        ).then(
            update_nkb_dropdown,
            [],
            rag_nkb_dropdown
        )

        def run_rag_model_task(model, rag_sys_prompt, selected_nkb, rag_prompt, temp, top_p, max_tok, gpu_l, port_val, oai_key, gem_key):
            try:
                if not selected_nkb:
                    return "오류: 지식 파일(.nkb)을 선택해주세요."
                
                nkb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb", selected_nkb)
                if not os.path.exists(nkb_path):
                    return f"오류: {selected_nkb} 파일이 존재하지 않습니다."
                
                with open(nkb_path, 'r', encoding='utf-8') as f:
                    kb_content = f.read().strip()
                
                full_sys_prompt = f"아래는 참고 문서입니다. 이 문서를 기반으로 질문에 답변하세요.\n---문서 시작---\n{kb_content}\n---문서 끝---\n\n{rag_sys_prompt}"
                benchmark.set_system_prompt(model, full_sys_prompt)
                
                result = benchmark.generate(model, rag_prompt, max_tok, temp, top_p, True, oai_key, gem_key, port_val, gpu_l)
                if 'error' in result:
                    return f"오류: {result['error']}"
                return result['output']
            except Exception as e:
                return f"실행 중 예외 발생: {e}"

        rag_run_button.click(
            run_rag_model_task,
            [rag_model_dropdown, rag_system_prompt_input, rag_nkb_dropdown, rag_prompt_input, temp, top_p, max_tokens, gpu_layers, port, openai_key, gemini_key],
            rag_output_text
        )

        def load_prompt_for_model(model_name):
            if not model_name: return "", "모델을 선택하세요."
            prompt = benchmark.get_system_prompt(model_name)
            return prompt, f"'{model_name}'의 프롬프트를 로드했습니다."
        
        load_prompt_button.click(load_prompt_for_model, manage_model_dropdown, [manage_system_prompt, system_prompt_status])

        def save_prompt_for_model(model_name, system_prompt, filename):
            if not model_name: return "모델을 선택하세요."
            benchmark.set_system_prompt(model_name, system_prompt)
            benchmark.save_system_prompt(model_name, filename.strip() or None)
            return f"'{model_name}'의 시스템 프롬프트를 저장했습니다."

        save_prompt_button.click(save_prompt_for_model, [manage_model_dropdown, manage_system_prompt, template_name], system_prompt_status)

        def load_template_content_and_update(template_name):
            content = benchmark.load_template_content(template_name)
            return content, f"'{template_name}' 템플릿을 로드했습니다."

        load_template_button.click(load_template_content_and_update, manage_template_dropdown, [manage_system_prompt, system_prompt_status])

        # --- Handlers for Single Test Tab Templates ---
        def load_template_for_single_test(template_name):
            return benchmark.load_template_content(template_name)

        single_system_prompt_template.change(
            load_template_for_single_test,
            single_system_prompt_template,
            single_system_prompt_input
        )
        rag_system_prompt_template.change(
            load_template_for_single_test,
            rag_system_prompt_template,
            rag_system_prompt_input
        )

        # --- Chatbot Event Handlers ---
        def load_template_for_chatbot(template_name):
            return benchmark.load_template_content(template_name)

        chatbot_system_prompt_template.change(
            load_template_for_chatbot,
            chatbot_system_prompt_template,
            chatbot_system_prompt_input
        )

        # Agent 템플릿 로딩 핸들러 추가
        def load_template_for_agent(template_name):
            if not template_name:
                return ""
            return benchmark.load_template_content(template_name)

        agent_system_prompt_template.change(
            load_template_for_agent,
            agent_system_prompt_template,
            agent_system_prompt_input
        )

        def toggle_chatbot_evaluation_ui(use_evaluation):
            """챗봇 출력 평가 옵션에 따라 관련 UI 요소들의 가시성을 조절합니다."""
            return gr.update(visible=use_evaluation), gr.update(visible=use_evaluation)

        chatbot_use_evaluation.change(
            toggle_chatbot_evaluation_ui,
            chatbot_use_evaluation,
            [chatbot_evaluation_model, chatbot_max_retries]
        )

        def send_message(message, history, model, system_prompt, temp, top_p, max_tokens, gpu_l, port_val, oai_key, gem_key, use_evaluation=False, eval_model="gpt-4o", max_retries=3, use_agent_dialogue=True, agent_model=None, agent_system_prompt="", agent_temp=0.3, agent_top_p=0.9, agent_max_tokens=256):
            global neo_executor, current_kb_file
            
            if not message.strip():
                return history, "", "메시지를 입력해주세요."
            
            try:
                # Agent 대화 모드가 활성화된 경우
                if use_agent_dialogue and agent_model:
                    print(f"[AGENT 모드] Agent와 사용자 대화 시작 (모델: {agent_model})")
                    
                    # Agent 시스템 프롬프트 로드
                    if not agent_system_prompt:
                        agent_system_prompt = benchmark.get_system_prompt(agent_model)
                    
                    # Agent와의 대화 기록 구성
                    agent_messages = []
                    if history:
                        # 기존 대화 기록을 Agent 대화로 변환
                        for user_msg, bot_msg in history:
                            agent_messages.append(f"사용자: {user_msg}")
                            agent_messages.append(f"Agent: {bot_msg}")
                    
                    # 현재 메시지 추가
                    agent_messages.append(f"사용자: {message}")
                    
                    # Agent에게 전달할 프롬프트 구성
                    agent_prompt = f"""{agent_system_prompt}

=== 대화 기록 ===
{chr(10).join(agent_messages)}

=== 지침 ===
위의 대화 기록을 참고하여 사용자의 국민건강보험 관련 정보가 충분한지 판단하고:

1. **정보가 충분한 경우**: 
   - "구조화완료:"로 시작하여 JSON 형태로 구조화된 정보를 제공
   - 예시: "구조화완료: {{"가입자유형": "직장가입자", "연소득": "30000000", ...}}"

2. **정보가 부족한 경우**: 
   - "추가질문:"으로 시작하여 국민건강보험 관련 추가 정보를 요청
   - 예시: "추가질문: 연소득을 알려주시면 보험료 산정이 정확해집니다."

3. **범위 외 질문인 경우**:
   - 국민건강보험과 무관한 질문(인사말, 잡담 등)은 "범위외:"로 시작하여 거절
   - 예시: "범위외: 저는 국민건강보험 관련 질문만 답변할 수 있습니다."

**중요**: 기술적 질문이나 시스템 관련 질문은 하지 마세요. 오직 국민건강보험 관련 정보 수집에만 집중하세요.

Agent:"""
                    
                    print(f"[AGENT] Agent 프롬프트 전송 중...")
                    agent_result = benchmark.generate(agent_model, agent_prompt, agent_max_tokens, agent_temp, agent_top_p, False, oai_key, gem_key, port_val, gpu_l)
                    
                    if 'error' in agent_result:
                        error_msg = f"Agent 처리 오류: {agent_result['error']}"
                        history.append((message, error_msg))
                        return history, "", error_msg
                    
                    agent_response = agent_result['output'].strip()
                    print(f"[AGENT 응답]\n{agent_response}")
                    
                    # Agent 응답 분석
                    if agent_response.startswith("구조화완료:"):
                        # 정보가 충분한 경우, 구조화된 정보 추출
                        structured_info = agent_response.replace("구조화완료:", "").strip()
                        print(f"[AGENT] 구조화된 정보 추출: {structured_info}")
                        
                        # 구조화된 정보를 NEO Query 변환에 사용
                        user_query_for_neo = f"사용자 상황: {message}\n구조화된 정보: {structured_info}"
                        
                        # Agent 응답을 대화 기록에 추가
                        history.append((message, agent_response))
                        
                        # NEO Query 변환으로 진행
                        print(f"[AGENT] NEO Query 변환 단계로 진행")
                        
                    elif agent_response.startswith("추가질문:"):
                        # 정보가 부족한 경우, Agent의 추가 질문을 그대로 반환
                        additional_question = agent_response.replace("추가질문:", "").strip()
                        history.append((message, additional_question))
                        return history, "", f"Agent 추가 질문 완료 ({len(additional_question)}자)"
                    
                    elif agent_response.startswith("범위외:"):
                        # 범위 외 질문인 경우, Agent의 거절 메시지를 그대로 반환
                        rejection_message = agent_response.replace("범위외:", "").strip()
                        history.append((message, rejection_message))
                        return history, "", f"Agent 범위 외 질문 거절 완료 ({len(rejection_message)}자)"
                    
                    else:
                        # 예상치 못한 응답 형식
                        history.append((message, agent_response))
                        return history, "", f"Agent 응답 완료 ({len(agent_response)}자)"
                
                else:
                    # Agent 모드가 비활성화된 경우, 기존 방식 사용
                    user_query_for_neo = message
                
                # 1. RAG: 질의 임베딩 → Chroma에서 유사 KB 검색
                # RAG KB 임베딩 캐시
                kb_embedding_cache = {}
                rag_context = ""
                if current_kb_file and CHROMA_RAG_AVAILABLE and rag_engine is not None:
                    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb", current_kb_file)
                    if os.path.exists(kb_path):
                        with open(kb_path, 'r', encoding='utf-8') as f:
                            kb_texts = [line.strip() for line in f if line.strip()]
                        # 최초 1회만 임베딩
                        if kb_path not in kb_embedding_cache:
                            rag_engine.build_kb(kb_texts)
                            kb_embedding_cache[kb_path] = True
                        retrieved_kb = rag_engine.query(user_query_for_neo, top_k=3)
                        rag_context = "\n".join(retrieved_kb)
                        print(f"[RAG 검색 결과]\n{rag_context}")
                elif current_kb_file and not CHROMA_RAG_AVAILABLE:
                    print("⚠️ Chroma RAG 기능이 비활성화되어 있습니다.")
                
                # 2. LLM에게 NEO Query 변환 요청 (RAG context 포함)
                neo_query_prompt = (
                    system_prompt +
                    "\n\n아래 KB(지식베이스) 내용을 반드시 참고하여, 사용자의 질의에 가장 적합한 NEO Query를 만들어라.\n"
                    "KB 내용과 최대한 일치하는 쿼리를 생성해야 하며, KB에 없는 정보는 생성하지 마라.\n"
                    "예시)\n"
                    "KB: (keep '(용어정의 근로자 '직업의_종류와_관계없이_근로의_대가로_보수를_받아_생활하는_사람으로서_공무원_및_교직원을_제외한_사람'))\n"
                    "질의: 근로자의 용어정의를 알려줘\n"
                    "→ 변환된 NEO Query: (keep '(용어정의 근로자 ?x))\n"
                    f"=== KB ===\n{rag_context}\n"
                    f"=== 질의 ===\n{user_query_for_neo}\n"
                    "=== 변환된 NEO Query만 출력하세요. ==="
                )
                print(f"[NEO Query 변환 프롬프트 미리보기]\n{neo_query_prompt[:200]}...")
                neo_query_result = benchmark.generate(model, neo_query_prompt, max_tokens=256, temperature=0.0, top_p=1.0, use_system_prompt=False, openai_api_key=oai_key, gemini_api_key=gem_key, port=port_val, gpu_layers=gpu_l)
                if 'error' in neo_query_result:
                    error_msg = f"NEO Query 변환 오류: {neo_query_result['error']}"
                    history.append((message, error_msg))
                    return history, "", error_msg
                neo_query = neo_query_result['output'].strip()
                print(f"[LLM이 변환한 NEO Query]\n{neo_query}")
                
                # 3. 변환된 NEO Query를 NEO Engine에 실행
                neo_result = ""
                if NEO_ENGINE_AVAILABLE and neo_executor is not None:
                    try:
                        print(f"NEO Engine에 쿼리 실행: {neo_query}")
                        result_code, neo_response = neo_executor.execute_query(neo_query)
                        if result_code == 1 and neo_response.strip():
                            neo_result = neo_response.strip()
                            print(f"[NEO Engine 응답]\n{neo_result}")
                        else:
                            print(f"NEO Engine 응답 없음 (코드: {result_code})")
                    except Exception as neo_error:
                        print(f"NEO Engine 처리 중 오류: {neo_error}")
                else:
                    print("NEO Engine이 초기화되지 않았거나 사용 불가")
                
                # 4. NEO Engine 결과를 LLM에 넣어 최종 답변 생성
                if neo_result.strip().lower() == 'nil' or not neo_result.strip():
                    response = '해당하는 정보가 없습니다.'
                    history.append((message, response))
                    base_status = "NEO Engine 결과 없음 (nil)"
                    status_msg = f"{base_status} ({len(response)}자)"
                    return history, "", status_msg

                final_prompt = system_prompt
                if current_kb_file:
                    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb", current_kb_file)
                    if os.path.exists(kb_path):
                        with open(kb_path, 'r', encoding='utf-8') as f:
                            kb_content = f.read().strip()
                        final_prompt += f"\n\n=== 지식 베이스 (KB) ===\n{kb_content}\n\n" \
                                        "=== 지침 ===\n위의 지식 베이스를 참고하여 답변하세요.\n"
                if neo_result:
                    final_prompt += (
                        f"\n\n=== NEO Engine 검색 결과 ===\n{neo_result}\n\n"
                        "=== 추가 지침 ===\n"
                        "위의 NEO Engine 검색 결과를 반드시 참고하여, 사람이 이해할 수 있는 자연어로 답변하세요.\n"
                        "NEO Engine 결과가 구조화된 형태(리스트, S-식 등)라면, 그 의미를 해석해서 자연어로 설명하세요.\n"
                        "예시)\n"
                        "NEO Engine 검색 결과: ((진료심사평가위원회 위원_연임_가능))\n"
                        "→ 답변: 진료심사평가위원회 위원은 연임이 가능합니다.\n"
                    )
                print(f"[최종 답변용 시스템 프롬프트 미리보기]\n{final_prompt[:200]}...")
                
                if use_evaluation:
                    print(f"평가 모드로 응답 생성 중... (모델: {model}, 평가 모델: {eval_model})")
                    benchmark.set_system_prompt(model, final_prompt)
                    result = generate_with_retry(benchmark, model, message, max_tokens, temp, top_p, 
                                               True, oai_key, gem_key, port_val, gpu_l, eval_model, max_retries, system_prompt_dir)
                else:
                    print(f"일반 모드로 응답 생성 중... (모델: {model})")
                    benchmark.set_system_prompt(model, final_prompt)
                    result = benchmark.generate(model, message, max_tokens, temp, top_p, True, oai_key, gem_key, port_val, gpu_l)
                
                if 'error' in result:
                    error_msg = f"오류: {result['error']}"
                    history.append((message, error_msg))
                    return history, "", error_msg
                
                response = result['output']
                history.append((message, response))
                
                # 응답 상태 메시지 생성
                if neo_result:
                    base_status = "NEO Query→NEO Engine→LLM 응답 완료"
                else:
                    base_status = "LLM 응답 완료 (NEO Engine 결과 없음)"
                
                if use_evaluation and 'attempts' in result:
                    status_msg = f"{base_status} ({len(response)}자, {result['attempts']}회 시도)"
                else:
                    status_msg = f"{base_status} ({len(response)}자)"
                
                return history, "", status_msg
                
            except Exception as e:
                error_msg = f"예외 발생: {str(e)}"
                history.append((message, error_msg))
                return history, "", error_msg

        def clear_chat():
            return [], "대화 기록이 지워졌습니다."

        def save_chat(history):
            if not history:
                return "저장할 대화가 없습니다."
            
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            
            # 대화 기록을 JSON 형식으로 저장
            chat_data = {
                "timestamp": timestamp,
                "messages": history
            }
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(chat_data, f, ensure_ascii=False, indent=2)
                return f"대화가 {filename}에 저장되었습니다."
            except Exception as e:
                return f"저장 실패: {str(e)}"

        def load_chat(file):
            if not file:
                return [], "파일을 선택해주세요."
            
            try:
                with open(file.name, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                
                if 'messages' in chat_data:
                    history = chat_data['messages']
                    return history, f"대화 기록을 불러왔습니다. ({len(history)}개 메시지)"
                else:
                    return [], "올바른 대화 파일 형식이 아닙니다."
                    
            except Exception as e:
                return [], f"파일 로드 실패: {str(e)}"

        def stop_generation():
            return "생성을 중지했습니다."

        # 챗봇 이벤트 연결
        send_btn.click(
            send_message,
            [chatbot_input, chatbot, chatbot_model_dropdown, chatbot_system_prompt_input, 
             chatbot_temp, chatbot_top_p, chatbot_max_tokens, gpu_layers, port, openai_key, gemini_key, chatbot_use_evaluation, chatbot_evaluation_model, chatbot_max_retries, use_agent_dialogue, agent_model_dropdown, agent_system_prompt_input, agent_temp, agent_top_p, agent_max_tokens],
            [chatbot, chatbot_input, chatbot_status]
        )
        
        chatbot_input.submit(
            send_message,
            [chatbot_input, chatbot, chatbot_model_dropdown, chatbot_system_prompt_input, 
             chatbot_temp, chatbot_top_p, chatbot_max_tokens, gpu_layers, port, openai_key, gemini_key, chatbot_use_evaluation, chatbot_evaluation_model, chatbot_max_retries, use_agent_dialogue, agent_model_dropdown, agent_system_prompt_input, agent_temp, agent_top_p, agent_max_tokens],
            [chatbot, chatbot_input, chatbot_status]
        )
        
        clear_chat_btn.click(
            clear_chat,
            [],
            [chatbot, chatbot_status]
        )
        
        save_chat_btn.click(
            save_chat,
            [chatbot],
            [chatbot_status]
        )
        
        load_chat_btn.click(
            lambda: gr.update(visible=True),
            [],
            [chat_file]
        )
        
        chat_file.change(
            load_chat,
            [chat_file],
            [chatbot, chatbot_status]
        ).then(
            lambda: gr.update(visible=False),
            [],
            [chat_file]
        )
        
        stop_btn.click(
            stop_generation,
            [],
            [chatbot_status]
        )

        demo.close(benchmark.server_manager.stop)

        def refresh_kb_files():
            """KB 파일 목록을 새로고침합니다."""
            kb_files = benchmark.get_available_kb_files()
            if kb_files:
                return gr.update(choices=kb_files, value=kb_files[0])
            else:
                return gr.update(choices=[], value=None)

        def preprocess_model(model_name, port_val, gpu_layers_val, selected_kb_file):
            """모델 서버를 시작하고 연결을 확인합니다."""
            global neo_executor, current_kb_file
            
            if not model_name:
                return "모델을 선택해주세요."
            
            try:
                # NEO Engine 초기화 (가능한 경우)
                neo_status = ""
                if NEO_ENGINE_AVAILABLE and neo_executor is None:
                    try:
                        print("NEO Engine 초기화 중...")
                        neo_executor = NEOExecutor()
                        
                        # 선택된 KB 파일 로드
                        kb_loaded = False
                        if selected_kb_file:
                            kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb", selected_kb_file)
                            if os.path.exists(kb_path):
                                if neo_executor.load_kb_file(kb_path):
                                    neo_status = f"✅ NEO Engine 초기화 완료 (KB: {selected_kb_file})"
                                    kb_loaded = True
                                    # 전역 변수에 선택된 KB 파일 저장
                                    current_kb_file = selected_kb_file
                                else:
                                    neo_status = f"❌ KB 파일 로드 실패: {selected_kb_file}"
                            else:
                                neo_status = f"❌ KB 파일을 찾을 수 없음: {selected_kb_file}"
                        else:
                            # 선택된 파일이 없으면 기본 경로들에서 시도
                            kb_paths = [
                                os.path.join(os.getcwd(), "facts.kb"),
                                os.path.join(os.getcwd(), "NEO", "facts.kb"),
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.kb"),
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), "NEO", "facts.kb")
                            ]
                            
                            for kb_path in kb_paths:
                                if os.path.exists(kb_path):
                                    if neo_executor.load_kb_file(kb_path):
                                        kb_filename = os.path.basename(kb_path)
                                        neo_status = f"✅ NEO Engine 초기화 완료 (KB: {kb_filename})"
                                        kb_loaded = True
                                        # 전역 변수에 선택된 KB 파일 저장
                                        current_kb_file = kb_filename
                                        break
                            
                            if not kb_loaded:
                                neo_status = "⚠️ NEO Engine 초기화 완료 (KB 파일 없음)"
                            
                    except Exception as neo_error:
                        neo_status = f"❌ NEO Engine 초기화 실패: {str(neo_error)}"
                        neo_executor = None
                elif NEO_ENGINE_AVAILABLE and neo_executor is not None:
                    neo_status = "✅ NEO Engine 이미 초기화됨"
                else:
                    neo_status = "⚠️ NEO Engine 사용 불가"
                
                # 모델 서버 시작
                model_status = ""
                if not model_name.startswith(("gpt-", "gemini")):
                    model_path = benchmark.model_paths.get(model_name)
                    print(f"로컬 모델 {model_name} 서버 시작 중... (경로: {model_path})")
                    if not model_path:
                        model_status = f"❌ {model_name} 경로를 찾을 수 없습니다."
                    elif benchmark.server_manager.start(model_path, benchmark.context_size, port_val, gpu_layers_val):
                        model_status = f"✅ {model_name} 서버 시작 완료 (포트: {port_val})"
                    else:
                        model_status = f"❌ {model_name} 서버 시작 실패"
                else:
                    model_status = f"✅ {model_name} API 모델 준비 완료"
                
                # 전체 상태 반환
                return f"{neo_status}\n{model_status}"
                    
            except Exception as e:
                return f"❌ 전처리 중 오류 발생: {str(e)}"

        # KB 파일 관련 이벤트 핸들러
        refresh_kb_btn.click(
            refresh_kb_files,
            [],
            [kb_file_dropdown]
        )
        
        preprocess_btn.click(
            preprocess_model,
            [chatbot_model_dropdown, port, gpu_layers, kb_file_dropdown],
            [preprocess_status]
        )

        # 결과 저장 및 시각화 핸들러 (벤치마크와 동일)
        querygen_save_button.click(
            save_and_plot,
            [querygen_results_json, querygen_viz_type],
            [querygen_save_status, querygen_summary_file, querygen_plot_output, querygen_summary_df_output]
        )
        querygen_viz_type.change(
            lambda file, choice: plot_summary(pd.read_csv(file.name), choice) if file else (None, pd.DataFrame()),
            [querygen_summary_file, querygen_viz_type],
            [querygen_plot_output, querygen_summary_df_output]
        )

        # --- Native 단일 모델 테스트 핸들러 ---
        def load_template_for_native_test(template_name):
            return benchmark.load_template_content(template_name)

        native_system_prompt_template.change(
            load_template_for_native_test,
            native_system_prompt_template,
            native_system_prompt_input
        )

        def run_native_model_task(model, sys_prompt, prompt, temp, top_p, max_tok, gpu_l, port_val, oai_key, gem_key):
            output = ""
            try:
                benchmark.set_system_prompt(model, sys_prompt)
                result = benchmark.generate(model, prompt, max_tok, temp, top_p, True, oai_key, gem_key, port_val, gpu_l)
                if 'error' in result:
                    return f"오류: {result['error']}"
                output = result['output']
                return output
            except Exception as e:
                return f"실행 중 예외 발생: {e}"

        # --- Native 단일 모델 테스트 핸들러 (대화형) ---
        def send_native_message(message, history, model, system_prompt, temp, top_p, max_tokens, gpu_l, port_val, oai_key, gem_key):
            output = ""
            if not message.strip():
                return history, "", "메시지를 입력하세요."
            try:
                # history를 OpenAI/chat 형식으로 변환
                chat_messages = []
                if system_prompt.strip():
                    chat_messages.append({"role": "system", "content": system_prompt.strip()})
                for q, a in history:
                    chat_messages.append({"role": "user", "content": q})
                    chat_messages.append({"role": "assistant", "content": a})
                chat_messages.append({"role": "user", "content": message.strip()})
                # 모델 호출
                json_data = {"messages": chat_messages, "max_tokens": max_tokens, "temperature": temp, "top_p": top_p}
                if model.startswith("gpt-"):
                    headers = {"Authorization": f"Bearer {oai_key}", "Content-Type": "application/json"}
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={"model": model, **json_data})
                    response.raise_for_status()
                    data = response.json()
                    output = data['choices'][0]['message']['content']
                elif model.startswith("gemini"):
                    if not gem_key:
                        return history, "", "Gemini API 키가 필요합니다."
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gem_key}"
                    full_prompt = system_prompt.strip() + "\n\n" + "\n".join([f"Q: {q}\nA: {a}" for q, a in history]) + f"\nQ: {message.strip()}\nA:"
                    json_data = {"contents": [{"parts": [{"text": full_prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temp, "topP": top_p}}
                    response = requests.post(url, json=json_data)
                    response.raise_for_status()
                    data = response.json()
                    output = data['candidates'][0]['content']['parts'][0]['text']
                else:
                    # 로컬 llama.cpp/ollama 등
                    model_type = benchmark.model_paths.get(model, "")
                    if model_type == "ollama":
                        # Ollama REST API로 바로 요청
                        url = "http://localhost:11434/api/generate"
                        json_data_ollama = {
                            "model": model,
                            "prompt": message,
                            "system": system_prompt,
                            "options": {
                                "temperature": temp,
                                "top_p": top_p,
                                "num_predict": max_tokens,
                                "stream": False
                            }
                        }
                        response = requests.post(url, json=json_data_ollama, timeout=120)
                        response.raise_for_status()
                        # 견고한 Ollama 응답 파싱: 빈 줄/공백/로그 무시, 여러 JSON 오브젝트 분리, done: true만 사용
                        import json
                        lines = response.text.splitlines()
                        final_response = ""
                        for line in lines:
                            line = line.strip()
                            if not line or not line.startswith("{"):
                                continue
                            json_chunks = []
                            if line.count("}{") > 0:
                                parts = line.replace('}{', '}|||{').split('|||')
                                json_chunks.extend(parts)
                            else:
                                json_chunks.append(line)
                            for chunk in json_chunks:
                                try:
                                    data = json.loads(chunk)
                                    if data.get("done"):
                                        final_response = data.get("response", "")
                                except Exception:
                                    continue
                        if not final_response:
                            # fallback: 마지막 줄의 response라도 사용
                            for line in reversed(lines):
                                line = line.strip()
                                if not line or not line.startswith("{"):
                                    continue
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        final_response = data["response"]
                                        break
                                except Exception:
                                    continue
                        output = final_response
                        history.append((message, output))
                        return history, "", f"응답 완료 ({len(output)}자)"
                    # llama.cpp 서버 실행 및 요청은 ollama가 아닐 때만
                    model_path = benchmark.model_paths.get(model)
                    if not model_path:
                        return history, "", f"모델 경로를 찾을 수 없습니다: {model}"
                    if not benchmark.server_manager.start(model_path, benchmark.context_size, port_val, gpu_l):
                        return history, "", f"로컬 서버({model}) 실행 실패 또는 연결 불가"
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(f"http://{SERVER_HOST}:{port_val}/v1/chat/completions", headers=headers, json=json_data)
                    response.raise_for_status()
                    # JSONDecodeError 방지: 여러 줄 중 첫 번째 유효한 JSON만 파싱
                    import json
                    data = None
                    try:
                        data = response.json()
                    except Exception:
                        lines = response.text.splitlines()
                        for line in lines:
                            try:
                                data = json.loads(line)
                                break
                            except Exception:
                                continue
                        if data is None:
                            raise  # 아무것도 파싱 안 되면 원래 에러 발생
                    output = data['choices'][0]['message']['content']
                history.append((message, output))
                return history, "", f"응답 완료 ({len(output)}자)"
            except Exception as e:
                error_msg = f"오류: {str(e)}"
                history.append((message, error_msg))
                return history, "", error_msg

        def clear_native_chat():
            return [], "대화 기록이 지워졌습니다."

        native_send_btn.click(
            send_native_message,
            [native_prompt_input, native_chatbot, native_model_dropdown, native_system_prompt_input, native_temp, native_top_p, native_max_tokens, gpu_layers, port, openai_key, gemini_key],
            [native_chatbot, native_prompt_input, native_status]
        )
        native_prompt_input.submit(
            send_native_message,
            [native_prompt_input, native_chatbot, native_model_dropdown, native_system_prompt_input, native_temp, native_top_p, native_max_tokens, gpu_layers, port, openai_key, gemini_key],
            [native_chatbot, native_prompt_input, native_status]
        )
        native_clear_btn.click(
            clear_native_chat,
            [],
            [native_chatbot, native_status]
        )

        # --- 시스템 프롬프트 템플릿 드롭다운 동적 표시 및 내용 로딩 핸들러 ---
        def toggle_sys_prompt_template_ui(checked):
            return gr.update(visible=checked), gr.update(visible=checked)

        use_sys_prompt.change(
            toggle_sys_prompt_template_ui,
            use_sys_prompt,
            [sys_prompt_template_dropdown, sys_prompt_template_content]
        )

        def load_sys_prompt_template_content(template_name):
            if not template_name:
                return ""
            path = os.path.join(system_prompt_dir, template_name)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return ""

        sys_prompt_template_dropdown.change(
            load_sys_prompt_template_content,
            sys_prompt_template_dropdown,
            sys_prompt_template_content
        )

        # --- 지식베이스 생성 탭 이벤트 핸들러 ---
        def toggle_kb_gen_input_ui(input_type):
            """입력 방식에 따라 UI 요소들의 가시성을 조절합니다."""
            if input_type == "텍스트 입력":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif input_type == "파일 업로드":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:  # URL 입력
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        kb_gen_input_type.change(
            toggle_kb_gen_input_ui,
            kb_gen_input_type,
            [kb_gen_text_input, kb_gen_file_input, kb_gen_url_input]
        )

        def load_template_for_kb_gen(template_name):
            """지식베이스 생성용 템플릿을 로드합니다."""
            if not template_name:
                return ""
            return benchmark.load_template_content(template_name)

        kb_gen_system_prompt_template.change(
            load_template_for_kb_gen,
            kb_gen_system_prompt_template,
            kb_gen_system_prompt_input
        )

        def extract_pdf_chunks_from_file(file, chunk_size=1200):
            """PDF 파일을 페이지별 또는 일정 길이로 분할하여 텍스트 chunk 리스트로 반환합니다."""
            try:
                file_path = file.name
                # PyPDF2 우선 사용
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        chunks = []
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                # 너무 길면 chunk_size 단위로 쪼갬
                                for i in range(0, len(text), chunk_size):
                                    chunk = text[i:i+chunk_size]
                                    if chunk.strip():
                                        chunks.append(chunk.strip())
                        return chunks
                except ImportError:
                    # pypdf fallback
                    try:
                        import pypdf
                        with open(file_path, 'rb') as f:
                            pdf_reader = pypdf.PdfReader(f)
                            chunks = []
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    for i in range(0, len(text), chunk_size):
                                        chunk = text[i:i+chunk_size]
                                        if chunk.strip():
                                            chunks.append(chunk.strip())
                            return chunks
                    except ImportError:
                        return ["PDF 파일을 읽으려면 PyPDF2 또는 pypdf 라이브러리가 필요합니다.\npip install PyPDF2 또는 pip install pypdf를 실행해주세요."]
            except Exception as e:
                return [f"PDF 파일 읽기 오류: {str(e)}"]

        def extract_text_from_file(file):
            """업로드된 파일에서 텍스트를 추출합니다."""
            if not file:
                return ""
            try:
                file_path = file.name
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                elif file_ext == '.md':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                elif file_ext == '.pdf':
                    # PDF는 분할 추출이 필요하므로 여기서는 전체 텍스트 반환 대신 chunk 리스트 반환
                    return extract_pdf_chunks_from_file(file)
                elif file_ext == '.docx':
                    try:
                        from docx import Document
                        doc = Document(file_path)
                        text = ""
                        for paragraph in doc.paragraphs:
                            text += paragraph.text + "\n"
                        return text
                    except ImportError:
                        return "DOCX 파일을 읽으려면 python-docx 라이브러리가 필요합니다. pip install python-docx를 실행해주세요."
                else:
                    return f"지원하지 않는 파일 형식입니다: {file_ext}"
            except Exception as e:
                return f"파일 읽기 오류: {str(e)}"

        def extract_text_from_url(url):
            """URL에서 텍스트를 추출합니다."""
            if not url or not url.strip():
                return ""
            
            try:
                import requests
                from bs4 import BeautifulSoup
                
                response = requests.get(url.strip(), timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 불필요한 태그 제거
                for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    tag.decompose()
                
                # 텍스트 추출
                text = soup.get_text()
                
                # 줄바꿈 정리
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # 최대 5000자로 제한
                
            except ImportError:
                return "URL에서 텍스트를 추출하려면 requests와 beautifulsoup4 라이브러리가 필요합니다. pip install requests beautifulsoup4를 실행해주세요."
            except Exception as e:
                return f"URL 텍스트 추출 오류: {str(e)}"

        def generate_knowledge_base(model, system_prompt, input_type, text_input, file_input, url_input, output_format, filename, temp, top_p, max_tokens, gpu_l, port_val, oai_key, gem_key):
            """지식베이스를 생성합니다."""
            try:
                # 입력 데이터 추출
                input_text = ""
                pdf_chunks = None
                if input_type == "텍스트 입력":
                    input_text = text_input.strip()
                elif input_type == "파일 업로드":
                    file_path = file_input.name if file_input else None
                    file_ext = os.path.splitext(file_path)[1].lower() if file_path else ""
                    if file_ext == '.pdf':
                        pdf_chunks = extract_pdf_chunks_from_file(file_input)
                    else:
                        input_text = extract_text_from_file(file_input)
                elif input_type == "URL 입력":
                    input_text = extract_text_from_url(url_input)
                
                if pdf_chunks is not None:
                    # PDF 분할 추출 케이스
                    all_kb_results = []
                    for idx, chunk in enumerate(pdf_chunks):
                        if not chunk.strip():
                            continue
                        # 출력 형식에 따른 프롬프트 구성
                        format_instructions = {
                            "NEO 형식 (.kb)": "NEO 엔진에서 사용할 수 있는 S-식 형태의 지식베이스를 생성하세요. 문서에서 발견되는 모든 중요한 지식 항목을 각각 별도의 (keep '(...)) 형태로 생성하세요. 예시:\n(keep '(용어정의 근로자 '직업의_종류와_관계없이_근로의_대가로_보수를_받아_생활하는_사람으로서_공무원_및_교직원을_제외한_사람'))\n(keep '(용어정의 건강보험 '질병이나부상으로인해발생한고액의진료비로가계에과도한부담이되는것을방지하기위한사회보장제도'))\n(keep '(건강보험_특성 의무적인보험가입및보험료납부))\n문서의 모든 관련 정보를 빠짐없이 추출하여 여러 개의 지식 항목으로 변환하세요.",
                            "자연어 형식 (.nkb)": "자연어로 된 지식베이스를 생성하세요. 문서에서 발견되는 모든 중요한 지식 항목을 각각 별도의 항목으로 작성하세요. 각 지식 항목을 명확하고 이해하기 쉽게 작성하세요.",
                            "JSON 형식 (.json)": "JSON 형태의 구조화된 지식베이스를 생성하세요. 문서의 모든 중요한 지식 항목을 concepts 배열에 포함하세요. 예시: {\"concepts\": [{\"term\": \"근로자\", \"definition\": \"직업의 종류와 관계없이 근로의 대가로 보수를 받아 생활하는 사람\"}, {\"term\": \"건강보험\", \"definition\": \"질병이나 부상으로 인해 발생한 고액의 진료비로 가계에 과도한 부담이 되는 것을 방지하기 위한 사회보장제도\"}]}"
                        }
                        format_instruction = format_instructions.get(output_format, "")
                        final_prompt = f"""{system_prompt}\n\n=== 입력 데이터 (PDF 분할 {idx+1}/{len(pdf_chunks)}) ===\n{chunk}\n\n=== 출력 형식 지침 ===\n{format_instruction}\n\n위의 입력 데이터를 바탕으로 {output_format}에 맞는 지식베이스를 생성하세요. \n중요: 문서에서 발견되는 모든 중요한 지식 항목을 빠짐없이 추출하여 여러 개의 지식 항목으로 변환하세요. \n하나의 지식 항목만 생성하지 말고, 문서에 포함된 모든 관련 정보를 각각 별도의 지식 항목으로 생성하세요."""
                        result = benchmark.generate(model, final_prompt, max_tokens, temp, top_p, True, oai_key, gem_key, port_val, gpu_l)
                        if 'error' in result:
                            continue
                        all_kb_results.append(result['output'])
                    # 결과 합치기 및 중복 제거
                    merged = '\n'.join(sorted(set('\n'.join(all_kb_results).splitlines())))
                    # 파일 저장
                    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
                    os.makedirs(kb_dir, exist_ok=True)
                    file_extensions = {
                        "NEO 형식 (.kb)": ".kb",
                        "자연어 형식 (.nkb)": ".nkb",
                        "JSON 형식 (.json)": ".json"
                    }
                    file_ext = file_extensions.get(output_format, ".kb")
                    file_path = os.path.join(kb_dir, f"{filename}{file_ext}")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(merged)
                    return f"지식베이스 생성 완료: {file_path}", merged
                else:
                    if not input_text:
                        return "지식베이스 생성을 기다리는 중...", "입력 데이터가 없습니다."
                    # 기존 단일 텍스트 처리 방식
                    format_instructions = {
                        "NEO 형식 (.kb)": "NEO 엔진에서 사용할 수 있는 S-식 형태의 지식베이스를 생성하세요. 문서에서 발견되는 모든 중요한 지식 항목을 각각 별도의 (keep '(...)) 형태로 생성하세요. 예시:\n(keep '(용어정의 근로자 '직업의_종류와_관계없이_근로의_대가로_보수를_받아_생활하는_사람으로서_공무원_및_교직원을_제외한_사람'))\n(keep '(용어정의 건강보험 '질병이나부상으로인해발생한고액의진료비로가계에과도한부담이되는것을방지하기위한사회보장제도'))\n(keep '(건강보험_특성 의무적인보험가입및보험료납부))\n문서의 모든 관련 정보를 빠짐없이 추출하여 여러 개의 지식 항목으로 변환하세요.",
                        "자연어 형식 (.nkb)": "자연어로 된 지식베이스를 생성하세요. 문서에서 발견되는 모든 중요한 지식 항목을 각각 별도의 항목으로 작성하세요. 각 지식 항목을 명확하고 이해하기 쉽게 작성하세요.",
                        "JSON 형식 (.json)": "JSON 형태의 구조화된 지식베이스를 생성하세요. 문서의 모든 중요한 지식 항목을 concepts 배열에 포함하세요. 예시: {\"concepts\": [{\"term\": \"근로자\", \"definition\": \"직업의 종류와 관계없이 근로의 대가로 보수를 받아 생활하는 사람\"}, {\"term\": \"건강보험\", \"definition\": \"질병이나 부상으로 인해 발생한 고액의 진료비로 가계에 과도한 부담이 되는 것을 방지하기 위한 사회보장제도\"}]}"
                    }
                    format_instruction = format_instructions.get(output_format, "")
                    final_prompt = f"""{system_prompt}\n\n=== 입력 데이터 ===\n{input_text}\n\n=== 출력 형식 지침 ===\n{format_instruction}\n\n위의 입력 데이터를 바탕으로 {output_format}에 맞는 지식베이스를 생성하세요. \n중요: 문서에서 발견되는 모든 중요한 지식 항목을 빠짐없이 추출하여 여러 개의 지식 항목으로 변환하세요. \n하나의 지식 항목만 생성하지 말고, 문서에 포함된 모든 관련 정보를 각각 별도의 지식 항목으로 생성하세요."""
                    result = benchmark.generate(model, final_prompt, max_tokens, temp, top_p, True, oai_key, gem_key, port_val, gpu_l)
                    if 'error' in result:
                        return "지식베이스 생성을 기다리는 중...", f"생성 오류: {result['error']}"
                    generated_kb = result['output']
                    # 파일 저장
                    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
                    os.makedirs(kb_dir, exist_ok=True)
                    file_extensions = {
                        "NEO 형식 (.kb)": ".kb",
                        "자연어 형식 (.nkb)": ".nkb", 
                        "JSON 형식 (.json)": ".json"
                    }
                    file_ext = file_extensions.get(output_format, ".kb")
                    file_path = os.path.join(kb_dir, f"{filename}{file_ext}")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(generated_kb)
                    return f"지식베이스 생성 완료: {file_path}", generated_kb
            except Exception as e:
                return "지식베이스 생성을 기다리는 중...", f"예외 발생: {str(e)}"

        def save_kb_generated_content(content, filename, output_format):
            """생성된 지식베이스를 파일로 저장합니다."""
            if not content.strip():
                return "저장할 내용이 없습니다."
            
            try:
                kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
                os.makedirs(kb_dir, exist_ok=True)
                
                file_extensions = {
                    "NEO 형식 (.kb)": ".kb",
                    "자연어 형식 (.nkb)": ".nkb",
                    "JSON 형식 (.json)": ".json"
                }
                
                file_ext = file_extensions.get(output_format, ".kb")
                file_path = os.path.join(kb_dir, f"{filename}{file_ext}")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return f"파일이 저장되었습니다: {file_path}"
                
            except Exception as e:
                return f"저장 실패: {str(e)}"

        def clear_kb_generated_content():
            """생성된 지식베이스 내용을 지웁니다."""
            return "지식베이스 생성을 기다리는 중...", ""

        def load_kb_generated_file(filename):
            """생성된 KB 파일을 로드합니다."""
            if not filename:
                return "파일을 선택해주세요."
            
            try:
                kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
                file_path = os.path.join(kb_dir, filename)
                
                if not os.path.exists(file_path):
                    return f"파일을 찾을 수 없습니다: {filename}"
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return f"파일 로드 완료: {filename} ({len(content)}자)"
                
            except Exception as e:
                return f"파일 로드 실패: {str(e)}"

        def refresh_kb_gen_file_list():
            """생성된 KB 파일 목록을 새로고침합니다."""
            kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
            kb_files = []
            
            if os.path.exists(kb_dir):
                for file in os.listdir(kb_dir):
                    if file.endswith(('.kb', '.nkb', '.json')):
                        kb_files.append(file)
            
            return gr.update(choices=sorted(kb_files), value=sorted(kb_files)[0] if kb_files else None)

        # 지식베이스 생성 이벤트 연결
        kb_gen_button.click(
            generate_knowledge_base,
            [kb_gen_model_dropdown, kb_gen_system_prompt_input, kb_gen_input_type, kb_gen_text_input, kb_gen_file_input, kb_gen_url_input, kb_gen_output_format, kb_gen_filename, kb_gen_temp, kb_gen_top_p, kb_gen_max_tokens, gpu_layers, port, openai_key, gemini_key],
            [kb_gen_progress, kb_gen_output]
        )
        
        kb_gen_save_btn.click(
            save_kb_generated_content,
            [kb_gen_output, kb_gen_filename, kb_gen_output_format],
            [kb_gen_save_status]
        )
        
        kb_gen_clear_btn.click(
            clear_kb_generated_content,
            [],
            [kb_gen_progress, kb_gen_output]
        )
        
        kb_gen_load_btn.click(
            load_kb_generated_file,
            [kb_gen_file_list],
            [kb_gen_save_status]
        )
        
        refresh_kb_gen_btn.click(
            refresh_kb_gen_file_list,
            [],
            [kb_gen_file_list]
        )

        # --- Native 단일 모델 테스트 핸들러 ---
        def load_template_for_native_test(template_name):
            return benchmark.load_template_content(template_name)

        native_system_prompt_template.change(
            load_template_for_native_test,
            native_system_prompt_template,
            native_system_prompt_input
        )

        def run_native_model_task(model, sys_prompt, prompt, temp, top_p, max_tok, gpu_l, port_val, oai_key, gem_key):
            output = ""
            try:
                benchmark.set_system_prompt(model, sys_prompt)
                result = benchmark.generate(model, prompt, max_tok, temp, top_p, True, oai_key, gem_key, port_val, gpu_l)
                if 'error' in result:
                    return f"오류: {result['error']}"
                output = result['output']
                return output
            except Exception as e:
                return f"실행 중 예외 발생: {e}"

        # --- Native 단일 모델 테스트 핸들러 (대화형) ---
        def send_native_message(message, history, model, system_prompt, temp, top_p, max_tokens, gpu_l, port_val, oai_key, gem_key):
            output = ""
            if not message.strip():
                return history, "", "메시지를 입력하세요."
            try:
                # history를 OpenAI/chat 형식으로 변환
                chat_messages = []
                if system_prompt.strip():
                    chat_messages.append({"role": "system", "content": system_prompt.strip()})
                for q, a in history:
                    chat_messages.append({"role": "user", "content": q})
                    chat_messages.append({"role": "assistant", "content": a})
                chat_messages.append({"role": "user", "content": message.strip()})
                # 모델 호출
                json_data = {"messages": chat_messages, "max_tokens": max_tokens, "temperature": temp, "top_p": top_p}
                if model.startswith("gpt-"):
                    headers = {"Authorization": f"Bearer {oai_key}", "Content-Type": "application/json"}
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={"model": model, **json_data})
                    response.raise_for_status()
                    data = response.json()
                    output = data['choices'][0]['message']['content']
                elif model.startswith("gemini"):
                    if not gem_key:
                        return history, "", "Gemini API 키가 필요합니다."
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gem_key}"
                    full_prompt = system_prompt.strip() + "\n\n" + "\n".join([f"Q: {q}\nA: {a}" for q, a in history]) + f"\nQ: {message.strip()}\nA:"
                    json_data = {"contents": [{"parts": [{"text": full_prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temp, "topP": top_p}}
                    response = requests.post(url, json=json_data)
                    response.raise_for_status()
                    data = response.json()
                    output = data['candidates'][0]['content']['parts'][0]['text']
                else:
                    # 로컬 llama.cpp/ollama 등
                    model_type = benchmark.model_paths.get(model, "")
                    if model_type == "ollama":
                        # Ollama REST API로 바로 요청
                        url = "http://localhost:11434/api/generate"
                        json_data_ollama = {
                            "model": model,
                            "prompt": message,
                            "system": system_prompt,
                            "options": {
                                "temperature": temp,
                                "top_p": top_p,
                                "num_predict": max_tokens,
                                "stream": False
                            }
                        }
                        response = requests.post(url, json=json_data_ollama, timeout=120)
                        response.raise_for_status()
                        # 견고한 Ollama 응답 파싱: 빈 줄/공백/로그 무시, 여러 JSON 오브젝트 분리, done: true만 사용
                        import json
                        lines = response.text.splitlines()
                        final_response = ""
                        for line in lines:
                            line = line.strip()
                            if not line or not line.startswith("{"):
                                continue
                            json_chunks = []
                            if line.count("}{") > 0:
                                parts = line.replace('}{', '}|||{').split('|||')
                                json_chunks.extend(parts)
                            else:
                                json_chunks.append(line)
                            for chunk in json_chunks:
                                try:
                                    data = json.loads(chunk)
                                    if data.get("done"):
                                        final_response = data.get("response", "")
                                except Exception:
                                    continue
                        if not final_response:
                            # fallback: 마지막 줄의 response라도 사용
                            for line in reversed(lines):
                                line = line.strip()
                                if not line or not line.startswith("{"):
                                    continue
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        final_response = data["response"]
                                        break
                                except Exception:
                                    continue
                        output = final_response
                        history.append((message, output))
                        return history, "", f"응답 완료 ({len(output)}자)"
                    # llama.cpp 서버 실행 및 요청은 ollama가 아닐 때만
                    model_path = benchmark.model_paths.get(model)
                    if not model_path:
                        return history, "", f"모델 경로를 찾을 수 없습니다: {model}"
                    if not benchmark.server_manager.start(model_path, benchmark.context_size, port_val, gpu_l):
                        return history, "", f"로컬 서버({model}) 실행 실패 또는 연결 불가"
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(f"http://{SERVER_HOST}:{port_val}/v1/chat/completions", headers=headers, json=json_data)
                    response.raise_for_status()
                    # JSONDecodeError 방지: 여러 줄 중 첫 번째 유효한 JSON만 파싱
                    import json
                    data = None
                    try:
                        data = response.json()
                    except Exception:
                        lines = response.text.splitlines()
                        for line in lines:
                            try:
                                data = json.loads(line)
                                break
                            except Exception:
                                continue
                        if data is None:
                            raise  # 아무것도 파싱 안 되면 원래 에러 발생
                    output = data['choices'][0]['message']['content']
                history.append((message, output))
                return history, "", f"응답 완료 ({len(output)}자)"
            except Exception as e:
                error_msg = f"오류: {str(e)}"
                history.append((message, error_msg))
                return history, "", error_msg

        def clear_native_chat():
            return [], "대화 기록이 지워졌습니다."

        native_send_btn.click(
            send_native_message,
            [native_prompt_input, native_chatbot, native_model_dropdown, native_system_prompt_input, native_temp, native_top_p, native_max_tokens, gpu_layers, port, openai_key, gemini_key],
            [native_chatbot, native_prompt_input, native_status]
        )
        native_prompt_input.submit(
            send_native_message,
            [native_prompt_input, native_chatbot, native_model_dropdown, native_system_prompt_input, native_temp, native_top_p, native_max_tokens, gpu_layers, port, openai_key, gemini_key],
            [native_chatbot, native_prompt_input, native_status]
        )
        native_clear_btn.click(
            clear_native_chat,
            [],
            [native_chatbot, native_status]
        )
    return demo

# =====================
# 6. main()
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--external_models_dir", default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads")
    parser.add_argument("--system_prompt_dir", default="./templates")
    parser.add_argument("--llama_server_path", default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    args = parser.parse_args()

    global LLAMA_SERVER_PATH
    LLAMA_SERVER_PATH = args.llama_server_path
    if not os.path.exists(LLAMA_SERVER_PATH):
        print(f"WARNING: llama-server not found at {LLAMA_SERVER_PATH}. Local models will not work.")

    model_paths = {
        "gpt-4o": "openai", "gpt-4-turbo": "openai", "gpt-3.5-turbo": "openai",
        "gemini-1.5-pro-latest": "google", "gemini-1.5-flash-latest": "google"
    }
    model_paths.update(find_gguf_files([args.models_dir, args.external_models_dir]))
    # ... Ollama 모델 자동탐색 등 기존 코드 유지 ...
    print("--- Loaded Models ---")
    for name in sorted(model_paths.keys()):
        print(f"- {name}")
    print("---------------------")
    print("[DEBUG] model_paths:", model_paths)
    demo = create_benchmark_interface(model_paths, system_prompt_dir=args.system_prompt_dir)
    demo.launch()

if __name__ == "__main__":
    main()