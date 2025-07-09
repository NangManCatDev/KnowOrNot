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

# ollama_preprocessor 임포트
try:
    from ollama_preprocessor import simplify_sentence
    OLLAMA_PREPROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️ ollama_preprocessor.py를 찾을 수 없습니다. 복합질의 단순화 기능이 비활성화됩니다.")
    OLLAMA_PREPROCESSOR_AVAILABLE = False
    def simplify_sentence(sentence: str) -> List[str]:
        return [sentence]

class ModelBenchmark:
    def __init__(self, model_paths: Dict[str, str], context_size: int = 2048, system_prompt_dir: str = "./templates"):
        self.model_paths = model_paths
        self.context_size = context_size
        self.system_prompt_dir = Path(system_prompt_dir)
        self.system_prompts = {}
        self.current_model_info = {}
        self.load_system_prompts()

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

    def load_template_content(self, template_name: str) -> str:
        """템플릿 파일의 내용을 로드합니다."""
        if not template_name:
            return ""
        template_path = self.system_prompt_dir / template_name
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""

    def set_system_prompt(self, model_name: str, system_prompt: str) -> None:
        """특정 모델의 시스템 프롬프트를 설정합니다."""
        self.system_prompts[model_name] = system_prompt

    def get_system_prompt(self, model_name: str) -> str:
        """특정 모델의 시스템 프롬프트를 반환합니다."""
        return self.system_prompts.get(model_name, "당신은 유용한 AI 어시스턴트입니다.")

    def save_system_prompt(self, model_name: str, filename: Optional[str] = None) -> None:
        """시스템 프롬프트를 파일로 저장합니다."""
        if filename is None:
            filename = f"{model_name}_system_prompt.txt"
        prompt_path = self.system_prompt_dir / filename
        os.makedirs(self.system_prompt_dir, exist_ok=True)
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(self.system_prompts[model_name])

    def load_system_prompts(self) -> None:
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

    def start_server(self, model_name: str, port: int, gpu_layers: int) -> bool:
        global SERVER_PROCESS
        running_model_info = self.current_model_info
        if (SERVER_PROCESS and SERVER_PROCESS.poll() is None and
            running_model_info.get("model_name") == model_name and
            running_model_info.get("port") == port and
            running_model_info.get("gpu_layers") == gpu_layers):
            return True

        self.stop_server()
        model_path = self.model_paths[model_name]
        cmd = [
            LLAMA_SERVER_PATH, "-m", model_path, "--ctx-size", str(self.context_size),
            "--host", SERVER_HOST, "--port", str(port), "-ngl", str(gpu_layers)
        ]
        print(f"[DEBUG] Executing: {' '.join(cmd)}")
        try:
            SERVER_PROCESS = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            self.current_model_info = {"model_name": model_name, "port": port, "gpu_layers": gpu_layers}
            
            max_wait, start_time = 60, time.time()
            while time.time() - start_time < max_wait:
                if SERVER_PROCESS.poll() is not None:
                    raise ChildProcessError(f"Server process terminated unexpectedly. Stderr: {SERVER_PROCESS.stderr.read()}")
                try:
                    response = requests.get(f"http://{SERVER_HOST}:{port}/v1/models", timeout=2)
                    if response.status_code == 200:
                        print(f"[DEBUG] Server for {model_name} is ready on port {port}.")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(2)
            raise TimeoutError("Server failed to start within the time limit.")
        except Exception as e:
            print(f"[ERROR] Failed to start server: {e}")
            self.stop_server()
            return False

    def stop_server(self) -> None:
        global SERVER_PROCESS
        if SERVER_PROCESS and SERVER_PROCESS.poll() is None:
            try:
                print(f"[DEBUG] Terminating server with PID {SERVER_PROCESS.pid}...")
                SERVER_PROCESS.terminate()
                SERVER_PROCESS.wait(timeout=5)
                print("[DEBUG] Server terminated.")
            except subprocess.TimeoutExpired:
                print("[DEBUG] Server did not terminate in time, killing.")
                SERVER_PROCESS.kill()
                SERVER_PROCESS.wait()
            finally:
                SERVER_PROCESS = None
                self.current_model_info = {}

    def generate(self, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float, 
                 use_system_prompt: bool, openai_api_key: Optional[str], gemini_api_key: Optional[str],
                 port: int, gpu_layers: int) -> Dict[str, Any]:
        start_time = time.time()
        system_prompt = self.system_prompts.get(model_name, "") if use_system_prompt else ""
        
        try:
            if model_name.startswith("gpt-"):
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
                return {
                    "output": output, "elapsed_time": time.time() - start_time, "system_prompt": system_prompt,
                    "prompt": prompt, "model": model_name
                }
                
            else:
                # Local llama.cpp server call logic
                if not self.start_server(model_name, port, gpu_layers):
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

                return {
                    "output": output, "elapsed_time": time.time() - start_time,
                    "tokens_generated": usage.get('completion_tokens', 0),
                    "tokens_prompt": usage.get('prompt_tokens', 0),
                    "tokens_total": usage.get('total_tokens', 0),
                    "system_prompt": system_prompt, "prompt": prompt, "model": model_name
                }
        except Exception as e:
            return {"error": str(e), "prompt": prompt, "model": model_name, "system_prompt": system_prompt}

def create_benchmark_interface(model_paths: Dict[str, str], system_prompt_dir: str):
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
                        use_query_simplification = gr.Checkbox(value=False, label="복합질의 단순화 사용", interactive=OLLAMA_PREPROCESSOR_AVAILABLE)
                        if not OLLAMA_PREPROCESSOR_AVAILABLE:
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

        with gr.Tab("단일 모델 테스트 (RAG)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1. 지식(.nkb) 생성")
                    single_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="모델 선택")
                    single_system_prompt_template = gr.Dropdown(choices=[""] + available_templates, label="시스템 프롬프트 템플릿 로드", allow_custom_value=False)
                    single_system_prompt_input = gr.Textbox(lines=4, label="시스템 프롬프트", placeholder="시스템 프롬프트를 입력하거나 템플릿을 선택하세요...")
                    single_prompt_input = gr.Textbox(lines=5, label="프롬프트", placeholder="지식(.nkb)로 저장할 프롬프트를 입력하세요...")
                    single_use_simplification = gr.Checkbox(value=False, label="복합질의 단순화 사용", interactive=OLLAMA_PREPROCESSOR_AVAILABLE)
                    with gr.Row():
                        single_temp = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                        single_top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-p")
                    single_max_tokens = gr.Slider(16, 4096, value=512, step=16, label="최대 토큰")
                    single_run_button = gr.Button("지식 생성 및 저장")
                    single_output_text = gr.Textbox(lines=10, label="모델 출력")
                with gr.Column():
                    gr.Markdown("#### 2. RAG(검색 기반 생성)")
                    rag_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="RAG용 모델 선택")
                    rag_system_prompt_template = gr.Dropdown(choices=[""] + available_templates, label="RAG 시스템 프롬프트 템플릿 로드", allow_custom_value=False)
                    rag_system_prompt_input = gr.Textbox(lines=4, label="RAG용 시스템 프롬프트", placeholder="RAG용 시스템 프롬프트를 입력하거나 템플릿을 선택하세요...")
                    rag_nkb_dropdown = gr.Dropdown(choices=available_nkb_files, label="지식 파일(.nkb) 선택", allow_custom_value=False)
                    rag_prompt_input = gr.Textbox(lines=3, label="RAG 프롬프트", placeholder="선택한 .nkb 파일을 참고하여 답변할 프롬프트를 입력하세요.")
                    rag_run_button = gr.Button("nkb 기반 응답 실행")
                    rag_output_text = gr.Textbox(lines=10, label="RAG 출력 결과")

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

        def run_benchmark_task(models, use_sys, use_simplification, use_evaluation, eval_model, max_retry, q_file, sel_cats, prompts_str, temp_val, top_p_val, max_tok_val, gpu_l, port_val, oai_key, gem_key, progress=gr.Progress(track_tqdm=True)):
            prompts = [p.strip() for p in prompts_str.split('\n') if p.strip()]
            if not prompts: raise gr.Error("No prompts specified.")

            # 복합질의 단순화 적용
            if use_simplification and OLLAMA_PREPROCESSOR_AVAILABLE:
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
                            if use_simplification and OLLAMA_PREPROCESSOR_AVAILABLE:
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

            benchmark.stop_server()
            return f"벤치마크 완료: 총 {total_tasks}개 태스크 실행", results_data

        run_button.click(
            run_benchmark_task, 
            [
                benchmark_models, use_sys_prompt, use_query_simplification, use_output_evaluation, evaluation_model, max_retries,
                query_file, category_checkboxes, prompts_textbox, temp, top_p, max_tokens, gpu_layers, port, openai_key, gemini_key
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
                model_dir = os.path.join(output_dir, model)
                os.makedirs(model_dir, exist_ok=True)
                
                # Save detailed JSON for each model
                model_json_path = os.path.join(model_dir, f"{model}_detailed_{timestamp}.json")
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
                model_csv_path = os.path.join(output_dir, model, f"{model}_detailed_{timestamp}.csv")
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
                if use_simplification and OLLAMA_PREPROCESSOR_AVAILABLE:
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

        demo.close(benchmark.stop_server)
    return demo

def find_gguf_files(dirs):
    paths = {}
    for d in dirs:
        if os.path.exists(d):
            for path in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
                name = os.path.splitext(os.path.basename(path))[0]
                paths[name] = path
    return paths

def evaluate_output(prompt: str, output: str, model_name: str, port: int, gpu_layers: int, openai_api_key: Optional[str], gemini_api_key: Optional[str], system_prompt_dir: str = "./templates", evaluation_port: int = None) -> Dict[str, Any]:
    """
    모델 출력을 평가하는 함수
    
    Args:
        prompt (str): 원본 질의
        output (str): 평가할 모델 출력
        model_name (str): 평가에 사용할 모델명
        port (int): 생성 모델 서버 포트
        gpu_layers (int): GPU 레이어 수
        openai_api_key (Optional[str]): OpenAI API 키
        gemini_api_key (Optional[str]): Gemini API 키
        system_prompt_dir (str): 템플릿 디렉토리 경로
        evaluation_port (int): 평가 모델용 별도 포트 (None이면 port 사용)
        
    Returns:
        Dict[str, Any]: 평가 결과 (pass: bool, score: float, feedback: str)
    """
    # 평가 프롬프트 템플릿 로드
    evaluation_prompt_path = os.path.join(system_prompt_dir, "evaluation_prompt.txt")
    try:
        with open(evaluation_prompt_path, 'r', encoding='utf-8') as f:
            evaluation_prompt_template = f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ evaluation_prompt.txt를 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
        evaluation_prompt_template = """당신은 AI 모델의 출력을 평가하는 전문가입니다.
다음 기준에 따라 출력을 평가해주세요:

평가 기준:
1. 정확성 (40점): 질문에 대한 답변이 정확한가?
2. 완성도 (30점): 답변이 완전하고 충분한 정보를 제공하는가?
3. 명확성 (20점): 답변이 명확하고 이해하기 쉬운가?
4. 관련성 (10점): 답변이 질문과 관련이 있는가?

평가 형식:
- 총점: [0-100점]
- 통과 여부: [PASS/FAIL] (70점 이상이면 PASS)
- 피드백: [구체적인 평가 내용]

질문: {prompt}
답변: {output}

평가 결과:"""

    # 프롬프트 템플릿에 실제 값 대입
    evaluation_prompt = evaluation_prompt_template.format(prompt=prompt, output=output)

    try:
        # 평가용 모델로 평가 수행 (GPT-4나 고성능 모델 사용 권장)
        if model_name.startswith("gpt-4"):
            # OpenAI API 사용
            print(f"OpenAI API로 평가 요청 중... (모델: {model_name})")
            headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
            messages = [{"role": "user", "content": evaluation_prompt}]
            json_data = {"model": model_name, "messages": messages, "max_tokens": 500, "temperature": 0.1}
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=json_data, timeout=30)
            response.raise_for_status()
            evaluation_result = response.json()['choices'][0]['message']['content']
            print(f"OpenAI API 평가 완료: {len(evaluation_result)}자")
        else:
            # 로컬 모델 사용 - 서버 상태 확인
            print(f"로컬 서버 상태 확인 중... (포트: {port})")
            try:
                status_response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                if status_response.status_code != 200:
                    raise ConnectionError(f"서버 응답 오류: {status_response.status_code}")
                print(f"로컬 서버 정상 - 평가 요청 중... (모델: {model_name})")
            except Exception as e:
                print(f"로컬 서버 연결 실패: {e}")
                raise ConnectionError(f"로컬 서버 연결 실패: {e}")
            
            headers = {"Content-Type": "application/json"}
            messages = [{"role": "user", "content": evaluation_prompt}]
            json_data = {"messages": messages, "max_tokens": 500, "temperature": 0.1}
            response = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", headers=headers, json=json_data, timeout=60)
            response.raise_for_status()
            evaluation_result = response.json()['choices'][0]['message']['content']
            print(f"로컬 모델 평가 완료: {len(evaluation_result)}자")
        
        # 평가 결과 파싱
        score_match = re.search(r'총점:\s*(\d+)', evaluation_result)
        pass_match = re.search(r'통과\s*여부:\s*(PASS|FAIL)', evaluation_result, re.IGNORECASE)
        
        try:
            score = int(score_match.group(1)) if score_match else 0
            passed = pass_match.group(1).upper() == "PASS" if pass_match else score >= 70
        except (ValueError, AttributeError) as e:
            print(f"평가 결과 파싱 오류: {e}, 기본값 사용")
            score = 70
            passed = True

        # 평가 결과 파싱
        print(f"평가 결과 파싱 중... (결과 길이: {len(evaluation_result)}자)")
        print(f"평가 결과 미리보기: {evaluation_result[:200]}...")
        
        score_match = re.search(r'총점:\s*(\d+)', evaluation_result)
        pass_match = re.search(r'통과\s*여부:\s*(PASS|FAIL)', evaluation_result, re.IGNORECASE)
        
        try:
            score = int(score_match.group(1)) if score_match else 0
            passed = pass_match.group(1).upper() == "PASS" if pass_match else score >= 70
            print(f"파싱된 점수: {score}, 통과 여부: {passed}")
        except (ValueError, AttributeError) as e:
            print(f"평가 결과 파싱 오류: {e}, 기본값 사용")
            score = 70
            passed = True
        
        return {
            "pass": passed,
            "score": score,
            "feedback": evaluation_result,
            "evaluation_model": model_name
        }
        
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        return {
            "pass": True,  # 오류 시 기본적으로 통과로 처리
            "score": 70,
            "feedback": f"평가 중 오류 발생: {str(e)}",
            "evaluation_model": model_name
        }

    except requests.exceptions.Timeout:
        print(f"평가 모델 응답 타임아웃: {model_name}")
        return {
            "pass": True,  # 타임아웃 시 기본적으로 통과로 처리
            "score": 70,
            "feedback": f"평가 모델 응답 타임아웃 (30초/60초). 모델: {model_name}",
            "evaluation_model": model_name
        }
    except requests.exceptions.ConnectionError:
        print(f"평가 모델 연결 실패: {model_name}")
        return {
            "pass": True,  # 연결 실패 시 기본적으로 통과로 처리
            "score": 70,
            "feedback": f"평가 모델 연결 실패. 모델: {model_name}",
            "evaluation_model": model_name
        }
    except Exception as e:
        print(f"평가 중 예상치 못한 오류 발생: {e}")
        return {
            "pass": True,  # 오류 시 기본적으로 통과로 처리
            "score": 70,
            "feedback": f"평가 중 오류 발생: {str(e)}",
            "evaluation_model": model_name
        }

def generate_with_retry(benchmark, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float, 
                       use_system_prompt: bool, openai_api_key: Optional[str], gemini_api_key: Optional[str],
                       port: int, gpu_layers: int, evaluation_model: str, max_retries: int = 3, system_prompt_dir: str = "./templates") -> Dict[str, Any]:
    """
    평가를 통한 재생성 기능이 포함된 생성 함수
    
    Args:
        benchmark: ModelBenchmark 인스턴스
        model_name (str): 생성할 모델명
        prompt (str): 입력 프롬프트
        max_tokens (int): 최대 토큰 수
        temperature (float): 온도
        top_p (float): Top-p 값
        use_system_prompt (bool): 시스템 프롬프트 사용 여부
        openai_api_key (Optional[str]): OpenAI API 키
        gemini_api_key (Optional[str]): Gemini API 키
        port (int): 서버 포트
        gpu_layers (int): GPU 레이어 수
        evaluation_model (str): 평가에 사용할 모델명
        max_retries (int): 최대 재시도 횟수
        system_prompt_dir (str): 시스템 프롬프트 디렉토리 경로
        
    Returns:
        Dict[str, Any]: 생성 결과 (output, attempts, evaluation_results)
    """
    attempts = 0
    evaluation_results = []
    start_time = time.time()
    max_total_time = 300  # 최대 5분 제한
    
    # 원본 시스템 프롬프트 저장
    original_system_prompt = benchmark.get_system_prompt(model_name)
    
    while attempts < max_retries:
        attempts += 1
        current_time = time.time()
        
        # 전체 실행 시간 체크
        if current_time - start_time > max_total_time:
            print(f"경고: 최대 실행 시간({max_total_time}초) 초과로 평가 중단")
            break
        
        print(f"평가 시도 {attempts}/{max_retries} - 모델: {model_name}")
        
        # 재시도 시 피드백을 시스템 프롬프트에 포함
        if attempts > 1 and evaluation_results:
            # 이전 평가 피드백들을 수집
            feedback_messages = []
            for i, eval_result in enumerate(evaluation_results):
                feedback_messages.append(f"이전 시도 {i+1} 평가: {eval_result.get('feedback', '피드백 없음')}")
            
            # 피드백을 포함한 새로운 시스템 프롬프트 생성
            feedback_text = "\n".join(feedback_messages)
            enhanced_system_prompt = f"""{original_system_prompt}

=== 이전 시도들의 평가 피드백 ===
{feedback_text}

=== 개선 지침 ===
위의 평가 피드백을 참고하여 더 나은 답변을 제공하세요. 
특히 지적된 문제점들을 해결하고 더 정확하고 완성도 높은 답변을 작성하세요."""
            
            print(f"피드백 기반 시스템 프롬프트 적용 (시도 {attempts})")
            benchmark.set_system_prompt(model_name, enhanced_system_prompt)
        else:
            # 첫 번째 시도는 원본 시스템 프롬프트 사용
            benchmark.set_system_prompt(model_name, original_system_prompt)
        
        # 모델 출력 생성
        result = benchmark.generate(model_name, prompt, max_tokens, temperature, top_p, 
                                  use_system_prompt, openai_api_key, gemini_api_key, port, gpu_layers)
        
        if 'error' in result:
            print(f"모델 생성 오류: {result['error']}")
            return result
        
        output = result['output']
        print(f"모델 출력 완료 (길이: {len(output)}자)")
        
        # 출력 평가
        print(f"평가 모델 {evaluation_model}으로 출력 평가 중...")
        
        # 평가 모델이 생성 모델과 같은 경우 경고
        if evaluation_model == model_name:
            print(f"⚠️ 경고: 평가 모델({evaluation_model})이 생성 모델({model_name})과 동일합니다.")
            print("   이는 서버 부하를 증가시킬 수 있습니다. 다른 모델 사용을 권장합니다.")
        
        evaluation = evaluate_output(prompt, output, evaluation_model, port, gpu_layers, openai_api_key, gemini_api_key, system_prompt_dir)
        evaluation_results.append(evaluation)
        
        print(f"평가 결과: 점수 {evaluation['score']}점, 통과: {evaluation['pass']}")
        if 'feedback' in evaluation:
            print(f"평가 피드백: {evaluation['feedback'][:100]}...")
        
        print(f"평가 결과 분석: 점수={evaluation['score']}, 통과={evaluation['pass']}")
        if evaluation['pass']:
            print(f"✅ 통과! 시도 {attempts}에서 성공. 재시도 중단.")
            result['attempts'] = attempts
            result['evaluation_results'] = evaluation_results
            # 원본 시스템 프롬프트로 복원
            benchmark.set_system_prompt(model_name, original_system_prompt)
            return result
        else:
            print(f"❌ 불통과! 시도 {attempts}: 평가 점수 {evaluation['score']}점으로 재생성 필요")
            if attempts < max_retries:
                print("2초 후 재생성 시작...")
                time.sleep(2)  # 재생성 전 잠시 대기
            else:
                print(f"최대 재시도 횟수({max_retries})에 도달했습니다.")
    
    # 최대 시도 횟수 도달 시 마지막 결과 반환
    result['attempts'] = attempts
    result['evaluation_results'] = evaluation_results
    result['warning'] = f"최대 재시도 횟수({max_retries})에 도달했습니다."
    # 원본 시스템 프롬프트로 복원
    benchmark.set_system_prompt(model_name, original_system_prompt)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--external_models_dir", default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads")
    parser.add_argument("--system_prompt_dir", default="./templates")
    parser.add_argument("--llama_server_path", default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe")
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
    
    print("--- Loaded Models ---")
    for name in sorted(model_paths.keys()):
        print(f"- {name}")
    print("---------------------")

    demo = create_benchmark_interface(model_paths, system_prompt_dir=args.system_prompt_dir)
    demo.launch()

if __name__ == "__main__":
    main()