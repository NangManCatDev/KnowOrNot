import os
import time
import json
import argparse
import glob
import subprocess
import requests
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# llama.cpp 서버 관련 설정
LLAMA_SERVER_PATH = "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
API_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
SERVER_PROCESS = None

class ModelBenchmark:
    def __init__(self, model_paths: Dict[str, str], context_size: int = 2048, system_prompt_dir: str = "./templates"):
        """
        Initialize the benchmark with multiple models.
        
        Args:
            model_paths: Dictionary mapping model names to their file paths
            context_size: Maximum context size for the models
            system_prompt_dir: Directory containing system prompt templates
        """
        self.model_paths = model_paths
        self.context_size = context_size
        self.system_prompt_dir = Path(system_prompt_dir)
        self.system_prompts = {}
        self.current_model = None
        
        # 기본 시스템 프롬프트 로드
        self.load_system_prompts()
        
    def load_system_prompts(self) -> None:
        """시스템 프롬프트 템플릿을 로드합니다."""
        default_prompt_path = self.system_prompt_dir / "default_system_prompt.txt"
        
        # 기본 시스템 프롬프트 로드
        if default_prompt_path.exists():
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                default_prompt = f.read().strip()
        else:
            default_prompt = "당신은 유용한 AI 어시스턴트입니다. 정확하고 도움이 되는 답변을 제공하세요."
            
        # 모든 모델에 기본 시스템 프롬프트 적용
        for model_name in self.model_paths:
            self.system_prompts[model_name] = default_prompt
            
        # 모델별 시스템 프롬프트 찾기
        for model_name in self.model_paths:
            # 모델 이름에서 접두사 추출 (예: llama-2-7b -> llama)
            prefix = model_name.split('-')[0].lower()
            prompt_path = self.system_prompt_dir / f"{prefix}_system_prompt.txt"
            
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    self.system_prompts[model_name] = f.read().strip()
                    
            # 전체 모델 이름과 일치하는 파일이 있는지 확인
            specific_prompt_path = self.system_prompt_dir / f"{model_name}_system_prompt.txt"
            if specific_prompt_path.exists():
                with open(specific_prompt_path, 'r', encoding='utf-8') as f:
                    self.system_prompts[model_name] = f.read().strip()
    
    def get_available_templates(self) -> List[str]:
        """사용 가능한 모든 시스템 프롬프트 템플릿 파일 목록을 반환합니다."""
        templates = []
        if self.system_prompt_dir.exists():
            for file in self.system_prompt_dir.glob("*_system_prompt.txt"):
                templates.append(file.name)
        return sorted(templates)
    
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
    
    def start_server(self, model_name: str) -> bool:
        """
        지정된 모델로 llama-server를 시작합니다.
        
        Args:
            model_name: 사용할 모델 이름
            
        Returns:
            서버 시작 성공 여부
        """
        global SERVER_PROCESS
        
        # 이미 서버가 실행 중이고 같은 모델을 사용 중이면 그대로 사용
        if SERVER_PROCESS is not None and self.current_model == model_name:
            # 서버가 아직 실행 중인지 확인
            if SERVER_PROCESS.poll() is None:
                print(f"[DEBUG] 서버가 이미 실행 중입니다 ({model_name})...")
                return True
        
        # 기존 서버가 실행 중이면 중지
        self.stop_server()
        
        model_path = self.model_paths[model_name]
        
        # llama-server 명령어 구성
        cmd = [
            LLAMA_SERVER_PATH,
            "-m", model_path,
            "--ctx-size", str(self.context_size),
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--parallel", "1",  # 병렬 처리 수
            "-ngl", "100"       # 사용 가능한 GPU 레이어 수
        ]
        print(f"[DEBUG] llama.cpp 서버 실행 명령어: {' '.join(cmd)}")
        print(f"[DEBUG] 서버 실행 시각: 모델={model_name}")
        
        try:
            # 서버 실행
            SERVER_PROCESS = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            print(f"[DEBUG] llama-server 프로세스 PID: {SERVER_PROCESS.pid}")
            
            # 현재 실행 중인 모델 이름 저장
            self.current_model = model_name
            
            # 서버 시작 대기 및 준비 확인 (최대 60초)
            print(f"[DEBUG] 서버 준비 대기 (최대 60초)")
            max_wait = 60
            interval = 2
            waited = 0
            while waited < max_wait:
                try:
                    response = requests.get(f"{API_BASE_URL}/models", timeout=3)
                    if response.status_code == 200:
                        print(f"[DEBUG] 서버 API 연결 성공! (모델: {model_name})")
                        return True
                except Exception:
                    pass
                time.sleep(interval)
                waited += interval
            print(f"[DEBUG] 서버 시작 실패 (모델: {model_name})")
            self.stop_server()
            return False
            
        except Exception as e:
            print(f"[DEBUG] 서버 시작 오류: {str(e)} (모델: {model_name})")
            self.stop_server()
            return False
    
    def stop_server(self) -> None:
        """실행 중인 llama-server를 중지합니다."""
        global SERVER_PROCESS
        
        if SERVER_PROCESS is not None:
            try:
                print("[DEBUG] 서버 종료 시도...")
                # 서버 프로세스 종료
                SERVER_PROCESS.terminate()
                # 최대 5초 대기
                SERVER_PROCESS.wait(timeout=5)
                print("[DEBUG] 서버 정상 종료 완료.")
            except subprocess.TimeoutExpired:
                # 종료되지 않으면 강제 종료
                print("[DEBUG] 서버 강제 종료...")
                SERVER_PROCESS.kill()
                print("[DEBUG] 서버 강제 종료 완료.")
            SERVER_PROCESS = None
            self.current_model = None
            import time
            print("[DEBUG] 서버 종료 후 3초 대기...")
            time.sleep(3)
    
    def generate(self, model_name: str, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.7, top_p: float = 0.95, 
                 use_system_prompt: bool = True, 
                 openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        모델을 사용하여 텍스트를 생성합니다.
        (GPT, Gemini는 llama.cpp 서버를 사용하지 않고 API로 직접 요청)
        """
        import requests
        import time
        # 1. OpenAI GPT 계열
        if model_name.startswith("gpt-"):
            if not openai_api_key:
                raise Exception("OpenAI API 키가 필요합니다.")
            start_time = time.time()
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            }
            messages = []
            if use_system_prompt and hasattr(self, 'system_prompts') and model_name in self.system_prompts:
                messages.append({"role": "system", "content": self.system_prompts[model_name]})
            messages.append({"role": "user", "content": prompt})
            data = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            }
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code != 200:
                raise Exception(f"OpenAI API 오류: {response.status_code} - {response.text}")
            response_data = response.json()
            end_time = time.time()
            elapsed_time = end_time - start_time
            model_output = response_data['choices'][0]['message']['content']
            tokens_prompt = response_data.get('usage', {}).get('prompt_tokens', int(len(prompt.split()) * 1.3))
            tokens_generated = response_data.get('usage', {}).get('completion_tokens', int(len(model_output.split()) * 1.3))
            tokens_total = response_data.get('usage', {}).get('total_tokens', tokens_prompt + tokens_generated)
            return {
                "model": model_name,
                "prompt": prompt,
                "system_prompt": self.system_prompts[model_name] if use_system_prompt and hasattr(self, 'system_prompts') and model_name in self.system_prompts else "",
                "output": model_output,
                "elapsed_time": elapsed_time,
                "tokens_generated": tokens_generated,
                "tokens_prompt": tokens_prompt,
                "tokens_total": tokens_total
            }
        # 2. Gemini 계열
        elif model_name.startswith("gemini"):
            if not gemini_api_key:
                raise Exception("Gemini API 키가 필요합니다.")
            start_time = time.time()
            # 시스템 프롬프트를 프롬프트 앞에 합쳐서 전달
            if use_system_prompt and model_name in self.system_prompts:
                full_prompt = self.system_prompts[model_name].strip() + "\n" + prompt
            else:
                full_prompt = prompt
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            headers = {"Content-Type": "application/json"}
            params = {"key": gemini_api_key}
            data = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                }
            }
            response = requests.post(url, headers=headers, params=params, json=data)
            if response.status_code != 200:
                raise Exception(f"Gemini API 오류: {response.status_code} - {response.text}")
            response_data = response.json()
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Gemini 응답 파싱
            try:
                model_output = response_data['candidates'][0]['content']['parts'][0]['text']
            except Exception:
                model_output = str(response_data)
            tokens_generated = int(len(model_output.split()) * 1.3)
            tokens_prompt = int(len(prompt.split()) * 1.3)
            tokens_total = tokens_prompt + tokens_generated
            return {
                "model": model_name,
                "prompt": prompt,
                "system_prompt": self.system_prompts[model_name] if use_system_prompt and hasattr(self, 'system_prompts') and model_name in self.system_prompts else "",
                "output": model_output,
                "elapsed_time": elapsed_time,
                "tokens_generated": tokens_generated,
                "tokens_prompt": tokens_prompt,
                "tokens_total": tokens_total
            }
        # 3. 로컬 llama.cpp
        else:
            # 서버 시작
            if not self.start_server(model_name):
                raise Exception(f"llama-server 시작 실패: {model_name}")
            # 시스템 프롬프트 적용
            if use_system_prompt and model_name in self.system_prompts:
                system_prompt = self.system_prompts[model_name]
            else:
                system_prompt = ""
            start_time = time.time()
            api_url = f"{API_BASE_URL}/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            data = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            }
            try:
                response = requests.post(api_url, headers=headers, json=data)
                if response.status_code != 200:
                    raise Exception(f"API 요청 오류: {response.status_code} - {response.text}")
                response_data = response.json()
                end_time = time.time()
                elapsed_time = end_time - start_time
                model_output = response_data['choices'][0]['message']['content']
                tokens_prompt = response_data.get('usage', {}).get('prompt_tokens', int(len(prompt.split()) * 1.3))
                tokens_generated = response_data.get('usage', {}).get('completion_tokens', int(len(model_output.split()) * 1.3))
                tokens_total = response_data.get('usage', {}).get('total_tokens', tokens_prompt + tokens_generated)
                result = {
                    "model": model_name,
                    "prompt": prompt,
                    "system_prompt": system_prompt if use_system_prompt else "",
                    "output": model_output,
                    "elapsed_time": elapsed_time,
                    "tokens_generated": tokens_generated,
                    "tokens_prompt": tokens_prompt,
                    "tokens_total": tokens_total
                }
                # 결과를 neo_benchmark/test.nkb로 자동 저장 (출력만 텍스트로 저장)
                save_dir = os.path.join(os.path.dirname(__file__), "test.nkb")
                with open(save_dir, 'w', encoding='utf-8') as f:
                    f.write(str(result["output"]))
                return result
            except Exception as e:
                raise Exception(f"API 요청 오류: {str(e)}")
    
    def benchmark(self, prompts: List[str], model_names: Optional[List[str]] = None, 
                 use_system_prompt: bool = True) -> Dict[str, Any]:
        """
        여러 프롬프트에 대해 모델 벤치마크를 실행합니다.
        
        Args:
            prompts: 테스트할 프롬프트 목록
            model_names: 벤치마크할 모델 이름 목록 (None이면 모든 모델)
            use_system_prompt: 시스템 프롬프트 사용 여부
            
        Returns:
            벤치마크 결과를 포함하는 딕셔너리
        """
        if model_names is None:
            model_names = list(self.model_paths.keys())
        
        results = {model_name: [] for model_name in model_names}
        
        for model_name in model_names:
            for prompt in prompts:
                result = self.generate(model_name, prompt, use_system_prompt=use_system_prompt)
                results[model_name].append(result)
        
        # 결과 집계
        summary = {}
        for model_name in model_names:
            model_results = results[model_name]
            total_time = sum(r["elapsed_time"] for r in model_results)
            total_tokens = sum(r["tokens_generated"] for r in model_results)
            
            summary[model_name] = {
                "total_time": total_time,
                "avg_time_per_prompt": total_time / len(prompts),
                "total_tokens": total_tokens,
                "avg_tokens_per_prompt": total_tokens / len(prompts),
                "tokens_per_second": total_tokens / total_time if total_time > 0 else 0
            }
        
        self.results = {
            "detailed": results,
            "summary": summary
        }
        
        return self.results
    
    def save_results(self, output_file: str) -> None:
        """벤치마크 결과를 JSON 파일로 저장합니다."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)


# Gradio 인터페이스 구현
def create_benchmark_interface(model_paths: Dict[str, str], system_prompt_dir: str = "./templates"):
    benchmark = ModelBenchmark(model_paths, system_prompt_dir=system_prompt_dir)
    
    # 사용 가능한 템플릿 목록 가져오기
    available_templates = benchmark.get_available_templates()
    
    # 2. API 키 관리 탭 추가
    openai_api_key_state = gr.State("")
    gemini_api_key_state = gr.State("")
    
    with gr.Blocks(title="sLLM 벤치마크") as demo:
        gr.Markdown("# sLLM 모델 벤치마크")
        gr.Markdown("llama.cpp를 사용한 소형 언어 모델 벤치마크 도구입니다.")
        
        with gr.Tab("단일 모델 테스트"):
            with gr.Row():
                # 왼쪽: .nkb 생성용
                with gr.Column():
                    gr.Markdown("#### 1. 지식(.nkb) 생성")
                    model_dropdown = gr.Dropdown(
                        choices=list(model_paths.keys()),
                        label="모델 선택"
                    )
                    # 시스템 프롬프트 템플릿 선택 드롭다운 추가
                    system_prompt_template = gr.Dropdown(
                        choices=[""] + available_templates,
                        value="",
                        label="시스템 프롬프트 템플릿",
                        allow_custom_value=False
                    )
                    system_prompt_input = gr.Textbox(
                        lines=4,
                        label="시스템 프롬프트",
                        placeholder="시스템 프롬프트를 입력하세요..."
                    )
                    prompt_input = gr.Textbox(
                        lines=5,
                        label="프롬프트",
                        placeholder="지식(.nkb)로 저장할 프롬프트를 입력하세요..."
                    )
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                            label="Temperature"
                        )
                        top_p = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.95, step=0.05,
                            label="Top-p"
                        )
                    max_tokens = gr.Slider(
                        minimum=16, maximum=2048, value=512, step=16,
                        label="최대 토큰 수"
                    )
                    run_button = gr.Button("지식 생성 및 저장")
                    output_text = gr.Textbox(lines=10, label="모델 출력")

                # 오른쪽: RAG
                with gr.Column():
                    gr.Markdown("#### 2. RAG(검색 기반 생성)")
                    rag_model_dropdown = gr.Dropdown(
                        choices=list(model_paths.keys()),
                        label="RAG용 모델 선택",
                        value=list(model_paths.keys())[0]
                    )
                    # RAG용 시스템 프롬프트 템플릿 선택 드롭다운 추가
                    rag_system_prompt_template = gr.Dropdown(
                        choices=[""] + available_templates,
                        value="",
                        label="RAG용 시스템 프롬프트 템플릿",
                        allow_custom_value=False
                    )
                    rag_system_prompt_input = gr.Textbox(
                        lines=4,
                        label="RAG용 시스템 프롬프트",
                        placeholder="RAG용 시스템 프롬프트를 입력하세요..."
                    )
                    rag_prompt_input = gr.Textbox(
                        lines=3,
                        label="RAG 프롬프트",
                        placeholder="test.nkb를 참고하여 답변할 프롬프트를 입력하세요."
                    )
                    rag_run_button = gr.Button("nkb 기반 응답 실행")
                    rag_output_text = gr.Textbox(lines=10, label="RAG 출력 결과")

            # 왼쪽: .nkb 생성 및 저장 함수
            def run_single_model(model_name, system_prompt, prompt, max_tokens, temperature, top_p):
                try:
                    benchmark.set_system_prompt(model_name, system_prompt)
                    result = benchmark.generate(
                        model_name, prompt, max_tokens, temperature, top_p, True
                    )
                    # 결과를 neo_benchmark/test.nkb로 자동 저장 (출력만 텍스트로 저장)
                    save_dir = os.path.join(os.path.dirname(__file__), "test.nkb")
                    with open(save_dir, 'w', encoding='utf-8') as f:
                        f.write(str(result["output"]))
                    return result["output"]
                except Exception as e:
                    return str(e)
            run_button.click(
                run_single_model,
                inputs=[model_dropdown, system_prompt_input, prompt_input, max_tokens, temperature, top_p],
                outputs=[output_text]
            )

            # 오른쪽: RAG 실행 함수
            def run_rag_model(rag_model, rag_system_prompt, rag_prompt):
                try:
                    # test.nkb 파일 읽기
                    test_nkb_path = os.path.join(os.path.dirname(__file__), "test.nkb")
                    if not os.path.exists(test_nkb_path):
                        return "test.nkb 파일이 존재하지 않습니다."
                    with open(test_nkb_path, 'r', encoding='utf-8') as f:
                        kb_content = f.read().strip()
                    # RAG용 시스템 프롬프트: 참고 문서 + 입력 시스템 프롬프트
                    rag_full_system_prompt = f"아래는 참고 문서입니다.\n{kb_content}\n\n{rag_system_prompt}"
                    benchmark.set_system_prompt(rag_model, rag_full_system_prompt)
                    result = benchmark.generate(
                        rag_model, rag_prompt, 512, 0.7, 0.95, True
                    )
                    return result["output"]
                except Exception as e:
                    return str(e)
            rag_run_button.click(
                run_rag_model,
                inputs=[rag_model_dropdown, rag_system_prompt_input, rag_prompt_input],
                outputs=[rag_output_text]
            )
        
        with gr.Tab("모델 비교 벤치마크"):
            with gr.Row():
                with gr.Column():
                    model_checkboxes = gr.CheckboxGroup(
                        choices=list(model_paths.keys()),
                        label="비교할 모델 선택",
                        value=list(model_paths.keys())
                    )
                    
                    # 벤치마크에서 시스템 프롬프트 사용 여부 옵션 추가
                    benchmark_system_prompt_checkbox = gr.Checkbox(
                        value=True,
                        label="시스템 프롬프트 사용"
                    )
                    
                    benchmark_prompts = gr.Textbox(
                        lines=8,
                        label="벤치마크 프롬프트 (한 줄에 하나씩)",
                        placeholder="벤치마크할 프롬프트를 입력하세요. 각 프롬프트는 새 줄로 구분합니다."
                    )
                    compare_button = gr.Button("벤치마크 실행")
                
                with gr.Column():
                    benchmark_output = gr.JSON(label="벤치마크 결과")
                    result_status = gr.Textbox(label="상태")
            
            def run_benchmark(models, use_system_prompt, prompts_text, openai_api_key_state, gemini_api_key_state):
                print("[DEBUG] run_benchmark 함수 진입 (벤치마크 실행 탭)")
                try:
                    prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                    if not prompts:
                        return {}, "프롬프트를 입력해주세요."
                    
                    results = benchmark.benchmark(prompts, models, use_system_prompt)
                    return results, f"{len(models)}개 모델, {len(prompts)}개 프롬프트로 벤치마크 완료"
                except Exception as e:
                    return {}, f"오류 발생: {str(e)}"
            
            compare_button.click(
                run_benchmark,
                inputs=[model_checkboxes, benchmark_system_prompt_checkbox, benchmark_prompts, openai_api_key_state, gemini_api_key_state],
                outputs=[benchmark_output, result_status]
            )
        
        # 시스템 프롬프트 관리 탭 추가
        with gr.Tab("시스템 프롬프트 관리"):
            with gr.Row():
                with gr.Column():
                    manage_model_dropdown = gr.Dropdown(
                        choices=list(model_paths.keys()),
                        label="모델 선택"
                    )
                    
                    # 템플릿 선택 드롭다운 추가
                    manage_template_dropdown = gr.Dropdown(
                        choices=[""] + available_templates,
                        value="",
                        label="템플릿 선택",
                        allow_custom_value=False
                    )
                    
                    # 템플릿 로드 버튼
                    load_template_button = gr.Button("템플릿 로드")
                    
                    manage_system_prompt = gr.Textbox(
                        lines=8,
                        label="시스템 프롬프트",
                        placeholder="시스템 프롬프트를 입력하세요..."
                    )
                    with gr.Row():
                        load_prompt_button = gr.Button("모델 프롬프트 로드")
                        save_prompt_button = gr.Button("저장")
                    
                    template_name = gr.Textbox(
                        label="템플릿 파일명 (저장 시 사용)",
                        placeholder="예: custom_system_prompt.txt"
                    )
                    
                with gr.Column():
                    system_prompt_status = gr.Textbox(label="상태")
                    available_templates_text = gr.Textbox(
                        label="사용 가능한 템플릿",
                        value="\n".join(available_templates) if available_templates else "사용 가능한 템플릿이 없습니다."
                    )
            
            # 모델 선택 시 해당 모델의 시스템 프롬프트 로드
            # manage_model_dropdown.change(
            #     load_system_prompt,
            #     inputs=[manage_model_dropdown],
            #     outputs=[manage_system_prompt]
            # )
            
            # 템플릿 로드
            load_template_button.click(
                inputs=[manage_template_dropdown],
                outputs=[manage_system_prompt, system_prompt_status]
            )
            
            # 시스템 프롬프트 로드 버튼
            def load_prompt_for_model(model_name):
                if model_name:
                    prompt = benchmark.get_system_prompt(model_name)
                    return prompt, f"'{model_name}'의 시스템 프롬프트를 로드했습니다."
                return "", "모델을 선택해주세요."
            
            load_prompt_button.click(
                load_prompt_for_model,
                inputs=[manage_model_dropdown],
                outputs=[manage_system_prompt, system_prompt_status]
            )
            
            # 시스템 프롬프트 저장 버튼
            def save_prompt_for_model(model_name, system_prompt, filename):
                if not model_name:
                    return "모델을 선택해주세요."
                
                benchmark.set_system_prompt(model_name, system_prompt)
                
                if filename:
                    benchmark.save_system_prompt(model_name, filename)
                    return f"'{model_name}'의 시스템 프롬프트를 '{filename}'로 저장했습니다."
                else:
                    benchmark.save_system_prompt(model_name)
                    return f"'{model_name}'의 시스템 프롬프트를 저장했습니다."
            
            save_prompt_button.click(
                save_prompt_for_model,
                inputs=[manage_model_dropdown, manage_system_prompt, template_name],
                outputs=[system_prompt_status]
            )
            
        with gr.Tab("API 키 관리"):
            openai_key = gr.Textbox(label="OpenAI API Key", type="password")
            gemini_key = gr.Textbox(label="Gemini API Key", type="password")
            save_button = gr.Button("저장")
            status = gr.Textbox(label="상태")
            def save_keys(openai, gemini):
                return "API 키가 저장되었습니다.", gr.update(value=openai), gr.update(value=gemini)
            save_button.click(
                save_keys,
                inputs=[openai_key, gemini_key],
                outputs=[status, openai_key, gemini_key]
            )
        
        with gr.Tab("벤치마크 실행"):
            with gr.Row():
                with gr.Column():
                    # 모델 선택
                    benchmark_models = gr.CheckboxGroup(
                        choices=list(model_paths.keys()),
                        label="벤치마크할 모델 선택",
                        value=list(model_paths.keys())
                    )
                    
                    # 시스템 프롬프트 사용 여부
                    benchmark_use_system_prompt = gr.Checkbox(
                        value=True,
                        label="시스템 프롬프트 사용"
                    )
                    
                    # 쿼리 파일 선택
                    query_file = gr.File(
                        label="쿼리 JSON 파일",
                        file_types=[".json"]
                    )
                    
                    # 카테고리 선택
                    category_checkboxes = gr.CheckboxGroup(
                        choices=[],
                        label="테스트할 카테고리 선택",
                        value=[]
                    )
                    
                    # 벤치마크 프롬프트 입력
                    benchmark_prompts = gr.Textbox(
                        lines=8,
                        label="벤치마크 프롬프트 (한 줄에 하나씩)",
                        placeholder="벤치마크할 프롬프트를 입력하세요. 각 프롬프트는 새 줄로 구분합니다."
                    )
                    
                    # 벤치마크 파라미터
                    with gr.Row():
                        benchmark_temperature = gr.Slider(
                            minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                            label="Temperature"
                        )
                        benchmark_top_p = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.95, step=0.05,
                            label="Top-p"
                        )
                    benchmark_max_tokens = gr.Slider(
                        minimum=16, maximum=2048, value=512, step=16,
                        label="최대 토큰 수"
                    )
                    
                    # 벤치마크 실행 버튼
                    run_benchmark_button = gr.Button("벤치마크 실행")
                
                with gr.Column():
                    # 진행 상황 표시
                    progress = gr.Textbox(
                        label="진행 상황",
                        value="대기 중..."
                    )
                    
                    # 결과 표시
                    benchmark_results = gr.JSON(label="벤치마크 결과")
                    
                    # 결과 저장 버튼
                    save_results_button = gr.Button("결과 저장")
                    
                    # 저장 상태 표시
                    save_status = gr.Textbox(label="저장 상태")
            
            def load_queries_from_json(file):
                if file is None:
                    return gr.update(choices=[], value=[]), ""
                try:
                    with open(file.name, 'r', encoding='utf-8') as f:
                        queries = json.load(f)
                    categories = list(queries.keys())
                    selected_categories = categories  # 기본값: 전체 선택
                    selected_queries = []
                    for category in selected_categories:
                        selected_queries.extend(queries[category])
                    return gr.update(choices=categories, value=selected_categories), "\n".join(selected_queries)
                except Exception as e:
                    return gr.update(choices=[], value=[]), f"쿼리 파일 로드 중 오류 발생: {str(e)}"
            
            def update_selected_queries(file, selected_categories):
                if file is None:
                    return ""
                try:
                    with open(file.name, 'r', encoding='utf-8') as f:
                        queries = json.load(f)
                    selected_queries = []
                    for category in selected_categories:
                        if category in queries:
                            selected_queries.extend(queries[category])
                    return "\n".join(selected_queries)
                except Exception as e:
                    return f"쿼리 업데이트 중 오류 발생: {str(e)}"
            
            # 쿼리 파일 로드 시 카테고리 목록 및 선택값 동시 업데이트
            query_file.change(
                load_queries_from_json,
                inputs=[query_file],
                outputs=[category_checkboxes, benchmark_prompts]
            )
            
            # 카테고리 선택 변경 시 선택된 쿼리 업데이트
            category_checkboxes.change(
                update_selected_queries,
                inputs=[query_file, category_checkboxes],
                outputs=[benchmark_prompts]
            )
            
            def run_benchmark(models, use_system_prompt, query_file_obj, selected_categories, prompts_text, temperature, top_p, max_tokens, openai_key, gemini_key):
                print("[DEBUG] run_benchmark 함수 진입 (벤치마크 실행 탭)")
                print(f"[DEBUG] models: {models}")
                print(f"[DEBUG] prompts_text: {prompts_text}")
                prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                print(f"[DEBUG] prompts: {prompts}")
                print(f"[DEBUG] selected_categories: {selected_categories}")
                try:
                    if not prompts:
                        return "프롬프트를 입력해주세요.", {}, "대기 중..."
                    # JSON 파일에서 카테고리 정보 로드
                    categories = {}
                    if query_file_obj is not None:
                        with open(query_file_obj.name, 'r', encoding='utf-8') as f:
                            categories = json.load(f)
                    progress_text = "벤치마크 시작...\n"
                    results = {}
                    for model in models:
                        print(f"[DEBUG] 모델 루프 진입: {model}")
                        progress_text += f"\n{model} 테스트 중..."
                        yield progress_text, results, "진행 중..."
                        model_results = {}
                        for category, category_queries in categories.items():
                            if category in selected_categories:
                                print(f"[DEBUG] 카테고리 루프 진입: {category}")
                                progress_text += f"\n  - {category} 카테고리 처리 중..."
                                yield progress_text, results, "진행 중..."
                                category_results = []
                                for query in category_queries:
                                    print(f"[DEBUG] generate 호출: model={model}, query={query}")
                                    try:
                                        result = benchmark.generate(
                                            model, query, max_tokens, temperature, top_p, use_system_prompt,
                                            openai_api_key=openai_key, gemini_api_key=gemini_key
                                        )
                                        result['category'] = category
                                        category_results.append(result)
                                    except Exception as e:
                                        import traceback
                                        print(traceback.format_exc())
                                        category_results.append({
                                            "error": str(e),
                                            "prompt": query,
                                            "model": model,
                                            "category": category
                                        })
                                model_results[category] = category_results
                        results[model] = model_results
                        progress_text += f"\n{model} 완료"
                    # 결과 요약
                    summary = {}
                    for model in models:
                        model_results = results[model]
                        model_summary = {}
                        
                        for category, category_results in model_results.items():
                            total_time = sum(r.get("elapsed_time", 0) for r in category_results)
                            total_tokens = sum(r.get("tokens_generated", 0) for r in category_results)
                            
                            model_summary[category] = {
                                "total_time": total_time,
                                "avg_time_per_prompt": total_time / len(category_results),
                                "total_tokens": total_tokens,
                                "avg_tokens_per_prompt": total_tokens / len(category_results),
                                "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
                                "accuracy": sum(1 for r in category_results if "error" not in r) / len(category_results)
                            }
                        
                        # 전체 요약
                        all_results = [r for category_results in model_results.values() for r in category_results]
                        total_time = sum(r.get("elapsed_time", 0) for r in all_results)
                        total_tokens = sum(r.get("tokens_generated", 0) for r in all_results)
                        
                        model_summary["전체"] = {
                            "total_time": total_time,
                            "avg_time_per_prompt": total_time / len(all_results),
                            "total_tokens": total_tokens,
                            "avg_tokens_per_prompt": total_tokens / len(all_results),
                            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
                            "accuracy": sum(1 for r in all_results if "error" not in r) / len(all_results)
                        }
                        
                        summary[model] = model_summary
                    
                    final_results = {
                        "detailed": results,
                        "summary": summary
                    }
                    
                    progress_text += "\n\n벤치마크 완료!"
                    yield progress_text, final_results, "완료"
                    
                except Exception as e:
                    yield f"오류 발생: {str(e)}", {}, "오류"
            
            def save_benchmark_results(results):
                if not results:
                    return "저장할 결과가 없습니다.", None
                try:
                    # 타임스탬프 생성
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    # 결과 저장 디렉토리 (항상 neo_benchmark/benchmark_results 하위)
                    output_dir = os.path.join(os.path.dirname(__file__), "benchmark_results", timestamp)
                    os.makedirs(output_dir, exist_ok=True)
                    # 시스템 프롬프트 필드 제거
                    def remove_system_prompt(obj):
                        if isinstance(obj, dict):
                            return {k: remove_system_prompt(v) for k, v in obj.items() if k != "system_prompt"}
                        elif isinstance(obj, list):
                            return [remove_system_prompt(v) for v in obj]
                        else:
                            return obj
                    results_no_sys_prompt = remove_system_prompt(results)

                    # 모델별로 detailed 결과 저장
                    for model, model_results in results["detailed"].items():
                        model_dir = os.path.join(output_dir, model)
                        os.makedirs(model_dir, exist_ok=True)
                        # 모델별 JSON 저장
                        model_json_path = os.path.join(model_dir, f"{model}_detailed_{timestamp}.json")
                        with open(model_json_path, 'w', encoding='utf-8') as f:
                            json.dump(remove_system_prompt(model_results), f, ensure_ascii=False, indent=2)
                        # 모델별 CSV 저장
                        csv_data = []
                        for category, category_results in model_results.items():
                            for result in category_results:
                                if "error" not in result:
                                    # 출력값에서 불필요한 줄바꿈/공백 제거
                                    output_clean = str(result["output"]).replace('\n', ' ').replace('  ', ' ').strip()
                                    csv_data.append({
                                        "모델": model,
                                        "카테고리": category,
                                        "프롬프트": result["prompt"],
                                        "출력": output_clean,
                                        "소요 시간(초)": round(result["elapsed_time"], 2),
                                        "생성 토큰": int(result["tokens_generated"]),
                                        "입력 토큰": int(result["tokens_prompt"]),
                                        "총 토큰": int(result["tokens_total"])
                                    })
                        if csv_data:
                            df = pd.DataFrame(csv_data, columns=["모델", "카테고리", "프롬프트", "출력", "소요 시간(초)", "생성 토큰", "입력 토큰", "총 토큰"])
                            csv_path = os.path.join(model_dir, f"{model}_detailed_{timestamp}.csv")
                            df.to_csv(csv_path, index=False, encoding='utf-8')

                    # 전체 summary 파일은 timestamp 폴더에 저장 (확장된 통계 포함)
                    summary_rows = []
                    for model, model_results in results["detailed"].items():
                        for category, category_results in model_results.items():
                            total = len(category_results)
                            success = sum(1 for r in category_results if "error" not in r)
                            fail = total - success
                            success_rate = success / total if total > 0 else 0
                            elapsed_times = [r["elapsed_time"] for r in category_results if "error" not in r]
                            tokens = [r["tokens_generated"] for r in category_results if "error" not in r]
                            tokens_per_sec = [r["tokens_generated"] / r["elapsed_time"] if r["elapsed_time"] > 0 else 0 for r in category_results if "error" not in r]
                            summary_rows.append({
                                "모델": model,
                                "카테고리": category,
                                "프롬프트 수": total,
                                "성공": success,
                                "실패": fail,
                                "성공률": round(success_rate * 100, 2),
                                "평균 처리 시간(초)": round(np.mean(elapsed_times), 2) if elapsed_times else 0,
                                "최대 처리 시간(초)": round(np.max(elapsed_times), 2) if elapsed_times else 0,
                                "최소 처리 시간(초)": round(np.min(elapsed_times), 2) if elapsed_times else 0,
                                "평균 생성 토큰": round(np.mean(tokens), 2) if tokens else 0,
                                "최대 생성 토큰": int(np.max(tokens)) if tokens else 0,
                                "최소 생성 토큰": int(np.min(tokens)) if tokens else 0,
                                "평균 토큰/초": round(np.mean(tokens_per_sec), 2) if tokens_per_sec else 0,
                                "최대 토큰/초": round(np.max(tokens_per_sec), 2) if tokens_per_sec else 0,
                                "최소 토큰/초": round(np.min(tokens_per_sec), 2) if tokens_per_sec else 0
                            })
                    summary_df = pd.DataFrame(summary_rows)
                    summary_csv_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
                    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
                    # 전체 summary JSON도 저장
                    summary_json_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")
                    with open(summary_json_path, 'w', encoding='utf-8') as f:
                        json.dump(remove_system_prompt(results["summary"]), f, ensure_ascii=False, indent=2)

                    return f"결과가 모델별로 저장되었습니다:\n{output_dir}", summary_csv_path
                except Exception as e:
                    return f"결과 저장 중 오류 발생: {str(e)}", None
            
            run_benchmark_button.click(
                run_benchmark,
                inputs=[
                    benchmark_models,
                    benchmark_use_system_prompt,
                    query_file,
                    category_checkboxes,
                    benchmark_prompts,
                    benchmark_temperature,
                    benchmark_top_p,
                    benchmark_max_tokens,
                    openai_key,
                    gemini_key
                ],
                outputs=[progress, benchmark_results, save_status]
            )
            
            # 벤치마크 시각화 탭
            with gr.Tab("벤치마크 시각화"):
                with gr.Row():
                    with gr.Column():
                        benchmark_file = gr.File(
                            label="벤치마크 결과 CSV 파일",
                            file_types=[".csv"],
                            file_count="multiple"
                        )
                        visualization_type = gr.Radio(
                            choices=[
                                "처리 시간 비교",
                                "토큰 생성 속도",
                                "생성된 토큰 수",
                                "정확도 비교",
                                "성공률",
                                "평균 처리 시간",
                                "평균 토큰/초",
                                "평균 생성 토큰"
                            ],
                            value="처리 시간 비교",
                            label="시각화 유형"
                        )
                        visualize_button = gr.Button("시각화")
                    with gr.Column():
                        visualization_output = gr.Plot(label="벤치마크 결과")
                        comparison_table = gr.Dataframe(
                            label="모델별 성능 비교",
                            headers=["모델", "정확도", "처리 시간", "토큰/초"]
                        )
                def visualize_benchmark_results(files, viz_type):
                    print(f'[DEBUG] 시각화 함수 진입, files={{files}}, viz_type={{viz_type}}')
                    comparison_data = []
                    if not files:
                        print('[DEBUG] 파일 없음')
                        return None, []
                    try:
                        # 여러 파일을 모두 읽어서 하나의 DataFrame으로 합침
                        dfs = []
                        for file in files:
                            df = pd.read_csv(file.name)
                            dfs.append(df)
                        df = pd.concat(dfs, ignore_index=True)
                        print(f'[DEBUG] CSV 병합 성공, shape={{df.shape}}')
                        # summary CSV인지 detailed CSV인지 자동 판별
                        is_summary = '성공률' in df.columns and '평균 처리 시간(초)' in df.columns
                        if is_summary:
                            # summary CSV 시각화 옵션
                            if viz_type == "성공/실패 건수":
                                plt.figure(figsize=(14, 8))
                                bar_width = 0.35
                                categories = df['카테고리'].unique()
                                x = np.arange(len(categories))
                                for idx, model in enumerate(df['모델'].unique()):
                                    model_data = df[df['모델'] == model]
                                    success = model_data['성공'].values
                                    fail = model_data['실패'].values
                                    plt.bar(x + idx * bar_width, success, width=bar_width, label=f"{model} - 성공")
                                    plt.bar(x + idx * bar_width, fail, width=bar_width, bottom=success, label=f"{model} - 실패", alpha=0.5, hatch='//')
                                plt.xticks(x + bar_width * (len(df['모델'].unique())-1)/2, categories, rotation=45)
                                plt.title('성공/실패 건수 (모델별, 카테고리별)')
                                plt.xlabel('카테고리')
                                plt.ylabel('건수')
                                plt.legend()
                            elif viz_type == "평균 처리 시간":
                                plt.figure(figsize=(12, 8))
                                for model in df['모델'].unique():
                                    model_data = df[df['모델'] == model]
                                    plt.bar(
                                        [f"{cat} ({model})" for cat in model_data['카테고리']],
                                        model_data['평균 처리 시간(초)'],
                                        label=model
                                    )
                                plt.title('Average Processing Time by Model and Category')
                                plt.xlabel('Category (Model)')
                                plt.ylabel('Average Processing Time (seconds)')
                            elif viz_type == "평균 토큰/초":
                                plt.figure(figsize=(12, 8))
                                for model in df['모델'].unique():
                                    model_data = df[df['모델'] == model]
                                    plt.bar(
                                        [f"{cat} ({model})" for cat in model_data['카테고리']],
                                        model_data['평균 토큰/초'],
                                        label=model
                                    )
                                plt.title('Average Tokens per Second by Model and Category')
                                plt.xlabel('Category (Model)')
                                plt.ylabel('Tokens/Second')
                            elif viz_type == "평균 생성 토큰":
                                plt.figure(figsize=(12, 8))
                                for model in df['모델'].unique():
                                    model_data = df[df['모델'] == model]
                                    plt.bar(
                                        [f"{cat} ({model})" for cat in model_data['카테고리']],
                                        model_data['평균 생성 토큰'],
                                        label=model
                                    )
                                plt.title('Average Generated Tokens by Model and Category')
                                plt.xlabel('Category (Model)')
                                plt.ylabel('Average Generated Tokens')
                            plt.xticks(rotation=45)
                            plt.legend()
                            plt.tight_layout()
                            # 비교표는 summary 전체
                            comparison_data = df.values.tolist()
                            return plt.gcf(), comparison_data
                        # detailed CSV 기존 시각화 옵션
                        if viz_type == "처리 시간 비교":
                            plt.figure(figsize=(12, 8))
                            for model in df['모델'].unique():
                                model_data = df[df['모델'] == model]
                                plt.bar(
                                    [f"{cat} ({model})" for cat in model_data['카테고리']],
                                    model_data['소요 시간(초)'],
                                    label=model
                                )
                            plt.title('Average Processing Time by Model and Category')
                            plt.xlabel('Category (Model)')
                            plt.ylabel('Average Processing Time (seconds)')
                        elif viz_type == "토큰 생성 속도":
                            plt.figure(figsize=(12, 8))
                            for model in df['모델'].unique():
                                model_data = df[df['모델'] == model]
                                tokens_per_second = model_data['생성 토큰'] / model_data['소요 시간(초)']
                                plt.bar(
                                    [f"{cat} ({model})" for cat in model_data['카테고리']],
                                    tokens_per_second,
                                    label=model
                                )
                            plt.title('Token Generation Speed by Model and Category')
                            plt.xlabel('Category (Model)')
                            plt.ylabel('Tokens/Second')
                        elif viz_type == "생성된 토큰 수":
                            plt.figure(figsize=(12, 8))
                            for model in df['모델'].unique():
                                model_data = df[df['모델'] == model]
                                plt.bar(
                                    [f"{cat} ({model})" for cat in model_data['카테고리']],
                                    model_data['생성 토큰'],
                                    label=model
                                )
                            plt.title('Average Generated Tokens by Model and Category')
                            plt.xlabel('Category (Model)')
                            plt.ylabel('Number of Generated Tokens')
                        elif viz_type == "정확도 비교":
                            plt.figure(figsize=(12, 8))
                            for model in df['모델'].unique():
                                model_data = df[df['모델'] == model]
                                accuracy = model_data.groupby('카테고리').apply(
                                    lambda x: sum(~x['출력'].str.contains('오류')) / len(x)
                                )
                                plt.bar(
                                    [f"{cat} ({model})" for cat in accuracy.index],
                                    accuracy.values,
                                    label=model
                                )
                            plt.title('Accuracy by Model and Category')
                            plt.xlabel('Category (Model)')
                            plt.ylabel('Accuracy')
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.tight_layout()
                        # 비교표 생성
                        for model in df['모델'].unique():
                            model_data = df[df['모델'] == model]
                            accuracy = sum(~model_data['출력'].str.contains('오류')) / len(model_data)
                            avg_time = model_data['소요 시간(초)'].mean()
                            avg_tokens = (model_data['생성 토큰'] / model_data['소요 시간(초)']).mean()
                            comparison_data.append([model, f"{accuracy:.2%}", f"{avg_time:.2f}초", f"{avg_tokens:.0f}"])
                        print(f'[DEBUG] comparison_data={{comparison_data}}')
                        return plt.gcf(), comparison_data
                    except Exception as e:
                        print(f'[시각화 오류] {{e}}')
                        return None, []
                # 기존 수동 시각화 버튼은 그대로 유지
                visualize_button.click(
                    visualize_benchmark_results,
                    inputs=[benchmark_file, visualization_type],
                    outputs=[visualization_output, comparison_table]
                )
                # save_and_visualize 함수 및 save_results_button.click 연결을 여기서 정의
                def save_and_visualize(results, viz_type):
                    status, summary_csv_path = save_benchmark_results(results)
                    if summary_csv_path and os.path.exists(summary_csv_path):
                        class DummyFile:
                            def __init__(self, name):
                                self.name = name
                        file_obj = DummyFile(summary_csv_path)
                        # 여러 파일을 리스트로 넘김
                        plot, table = visualize_benchmark_results([file_obj], viz_type)
                        return status, [summary_csv_path], plot, table
                    return status, None, None, None
                save_results_button.click(
                    save_and_visualize,
                    inputs=[benchmark_results, visualization_type],
                    outputs=[save_status, benchmark_file, visualization_output, comparison_table]
                )
            
        # Gradio 종료 시 서버 중지
        demo.load(lambda: None)
        demo.close(lambda: benchmark.stop_server())
    
    return demo

def find_model_files(directories):
    """여러 디렉토리에서 GGUF 모델 파일을 찾습니다."""
    model_paths = {}
    
    for directory in directories:
        if os.path.exists(directory):
            # 해당 디렉토리의 모든 .gguf 파일 찾기
            gguf_files = glob.glob(os.path.join(directory, "**", "*.gguf"), recursive=True)
            
            for model_path in gguf_files:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                model_paths[model_name] = model_path
    
    return model_paths

def main():
    parser = argparse.ArgumentParser(description="sLLM 모델 벤치마크")
    parser.add_argument("--models_dir", type=str, default="./models",
                        help="모델 파일이 저장된 디렉토리 경로")
    parser.add_argument("--external_models_dir", type=str, 
                        default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads",
                        help="외부 모델 파일이 저장된 디렉토리 경로")
    parser.add_argument("--system_prompt_dir", type=str, 
                        default="C:/Users/1/Desktop/workplace/code/KnowOrNot/neo_benchmark/templates",
                        help="시스템 프롬프트 템플릿이 저장된 디렉토리 경로")
    parser.add_argument("--llama_server_path", type=str, 
                        default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe",
                        help="llama-server 실행 파일 경로")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="서버 호스트 주소")
    parser.add_argument("--port", type=int, default=8080,
                        help="서버 포트 번호")
    args = parser.parse_args()
    
    # 서버 설정
    global LLAMA_SERVER_PATH, SERVER_HOST, SERVER_PORT, API_BASE_URL
    LLAMA_SERVER_PATH = args.llama_server_path
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    API_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
    
    # llama-server 실행 파일 존재 확인
    if not os.path.exists(LLAMA_SERVER_PATH):
        print(f"경고: llama-server 실행 파일을 찾을 수 없습니다: {LLAMA_SERVER_PATH}")
        print("올바른 경로를 지정해주세요.")
        return
    
    # Gemini 텍스트 모델 최신 공식 모델명만 남김 (404 발생 모델 제거)
    model_paths = {
        "gpt-3.5-turbo": "openai",
        "gpt-4": "openai",
        "gpt-4o": "openai",
        "gemini-1.5-pro-latest": "google",
        "gemini-1.5-flash-latest": "google",
        "gemini-1.0-pro": "google",
        "gemini-1.0-pro-001": "google",
    }
    # 2. 로컬 모델 자동 추가
    model_paths.update(find_model_files([args.models_dir, args.external_models_dir]))
    
    if not model_paths:
        # 예제 모델 경로 (사용자가 수정해야 함)
        model_paths = {
            "gpt-3.5-turbo": "openai",
            "gpt-4": "openai",
            "gpt-4o": "openai",
            "gemini-1.5-pro-latest": "google",
            "gemini-1.5-flash-latest": "google",
            "gemini-1.0-pro": "google",
            "gemini-1.0-pro-001": "google",
            "llama-2-7b-chat-q4_K_M": "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads/llama-2-7b-chat.Q4_K_M.gguf",
            "mistral-7b-instruct-q4_K_M": "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        }
        print(f"경고: 모델 디렉토리에서 .gguf 파일을 찾을 수 없습니다.")
        print("샘플 모델 경로를 사용합니다. 실제 모델 경로로 수정해주세요.")
    else:
        print(f"발견된 모델 수: {len(model_paths)}")
        for name, path in model_paths.items():
            print(f"  - {name}: {path}")
    
    # Gradio 인터페이스 생성 및 시작
    demo = create_benchmark_interface(model_paths, system_prompt_dir=args.system_prompt_dir)
    
    try:
        demo.launch(share=False)
    finally:
        # 종료 시 서버 중지 (추가 보호 장치)
        if 'benchmark' in locals():
            benchmark.stop_server()

if __name__ == "__main__":
    main() 