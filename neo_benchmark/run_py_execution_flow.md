# NEO 벤치마크 시스템 run.py 실행 과정 상세 분석

## 📋 목차
1. [실행 시작 및 초기화](#실행-시작-및-초기화)
2. [모듈 임포트 및 의존성 검사](#모듈-임포트-및-의존성-검사)
3. [전역 변수 및 설정 초기화](#전역-변수-및-설정-초기화)
4. [모델 탐색 및 경로 설정](#모델-탐색-및-경로-설정)
5. [Gradio UI 생성 과정](#gradio-ui-생성-과정)
6. [사용자 인터랙션 처리 과정](#사용자-인터랙션-처리-과정)
7. [각 탭별 실행 흐름](#각-탭별-실행-흐름)
8. [종료 및 정리 과정](#종료-및-정리-과정)

---

## 🚀 실행 시작 및 초기화

### **1. 스크립트 실행 시작**
```bash
python neo_benchmark/run.py
```

### **2. 명령행 인자 파싱**
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--external_models_dir", default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads")
    parser.add_argument("--system_prompt_dir", default="./templates")
    parser.add_argument("--llama_server_path", default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    args = parser.parse_args()
```

**실행 과정:**
1. **ArgumentParser 생성**: 명령행 인자를 처리할 파서 생성
2. **기본값 설정**: 각 인자의 기본값 정의
3. **인자 파싱**: 사용자가 입력한 명령행 인자 파싱

---

## 📦 모듈 임포트 및 의존성 검사

### **1. 기본 라이브러리 임포트**
```python
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
```

**실행 과정:**
1. **시스템 라이브러리 임포트**: os, time, json 등 기본 라이브러리
2. **타입 힌트 임포트**: typing 모듈에서 타입 힌트 클래스들
3. **경로 처리 라이브러리**: pathlib.Path 임포트
4. **시각화 라이브러리**: matplotlib 임포트
5. **플랫폼 정보**: platform 모듈 임포트

### **2. matplotlib 설정 (한글 폰트)**
```python
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False
```

**실행 과정:**
1. **플랫폼 감지**: Windows인지 다른 OS인지 확인
2. **폰트 설정**: Windows는 Malgun Gothic, 다른 OS는 NanumGothic
3. **마이너스 기호 설정**: 한글 마이너스 기호 깨짐 방지

### **3. Gradio 및 데이터 처리 라이브러리 임포트**
```python
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
```

**실행 과정:**
1. **라이브러리 임포트 시도**: gradio, numpy, pandas, matplotlib.pyplot 임포트
2. **ImportError 처리**: 임포트 실패 시 사용자에게 설치 안내
3. **프로그램 종료**: 필수 라이브러리가 없으면 프로그램 종료

### **4. NEO Engine 모듈 임포트**
```python
try:
    from engine import NEOExecutor
    NEO_ENGINE_AVAILABLE = True
    neo_executor = None
    current_kb_file = None
except ImportError:
    print("⚠️ engine.py를 찾을 수 없습니다. NEO Engine 기능이 비활성화됩니다.")
    NEO_ENGINE_AVAILABLE = False
    neo_executor = None
    current_kb_file = None
```

**실행 과정:**
1. **NEO Engine 임포트 시도**: engine.py에서 NEOExecutor 클래스 임포트
2. **성공 시**: NEO_ENGINE_AVAILABLE = True, 전역 변수 초기화
3. **실패 시**: NEO_ENGINE_AVAILABLE = False, 경고 메시지 출력

### **5. Chroma RAG 모듈 임포트**
```python
try:
    from chroma_rag import ChromaRAG
    rag_engine = ChromaRAG()
    CHROMA_RAG_AVAILABLE = True
except ImportError:
    print("⚠️ chroma_rag.py를 찾을 수 없습니다. Chroma RAG 기능이 비활성화됩니다.")
    CHROMA_RAG_AVAILABLE = False
    rag_engine = None
```

**실행 과정:**
1. **Chroma RAG 임포트 시도**: chroma_rag.py에서 ChromaRAG 클래스 임포트
2. **인스턴스 생성**: ChromaRAG() 인스턴스 생성
3. **실패 시**: CHROMA_RAG_AVAILABLE = False, 경고 메시지 출력

---

## ⚙️ 전역 변수 및 설정 초기화

### **1. 상수 설정**
```python
SYSTEM_PROMPT_DIR = "./templates"
QUERY_SIMPLIFICATION_AVAILABLE = True
LLAMA_SERVER_PATH = "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe"
SERVER_HOST = "127.0.0.1"
SERVER_PROCESS = None
```

**실행 과정:**
1. **시스템 프롬프트 디렉토리**: 템플릿 파일들이 저장된 경로 설정
2. **복합질의 단순화 플래그**: ollama_preprocessor.py 사용 가능 여부
3. **llama.cpp 서버 경로**: 로컬 모델 서버 실행 파일 경로
4. **서버 호스트**: 로컬 서버 호스트 주소
5. **서버 프로세스**: 현재 실행 중인 서버 프로세스 추적

---

## 🔍 모델 탐색 및 경로 설정

### **1. GGUF 모델 파일 탐색**
```python
def find_gguf_files(dirs: List[str]) -> Dict[str, str]:
    """지정된 디렉토리에서 gguf 모델 파일을 탐색"""
    paths = {}
    for d in dirs:
        if os.path.exists(d):
            for path in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
                name = os.path.splitext(os.path.basename(path))[0]
                paths[name] = path
    print(f"[DEBUG][find_gguf_files] 반환값: {paths}")
    return paths
```

**실행 과정:**
1. **디렉토리 순회**: 지정된 디렉토리들을 하나씩 확인
2. **존재 여부 확인**: 각 디렉토리가 실제로 존재하는지 확인
3. **재귀적 검색**: glob.glob을 사용하여 하위 디렉토리까지 검색
4. **파일명 추출**: .gguf 확장자를 제외한 파일명 추출
5. **딕셔너리 구성**: {모델명: 파일경로} 형태로 딕셔너리 구성

### **2. 모델 경로 딕셔너리 구성**
```python
model_paths = {
    "gpt-4o": "openai", 
    "gpt-4-turbo": "openai", 
    "gpt-3.5-turbo": "openai",
    "gemini-1.5-pro-latest": "google", 
    "gemini-1.5-flash-latest": "google"
}
model_paths.update(find_gguf_files([args.models_dir, args.external_models_dir]))
```

**실행 과정:**
1. **API 모델 설정**: OpenAI와 Google API 모델들 미리 설정
2. **로컬 모델 탐색**: find_gguf_files 함수로 로컬 GGUF 모델들 탐색
3. **딕셔너리 병합**: API 모델과 로컬 모델을 하나의 딕셔너리로 병합
4. **모델 목록 출력**: 찾은 모든 모델들의 목록을 콘솔에 출력

---

## 🎨 Gradio UI 생성 과정

### **1. create_benchmark_interface 함수 호출**
```python
demo = create_benchmark_interface(model_paths, system_prompt_dir=args.system_prompt_dir)
```

**실행 과정:**
1. **함수 호출**: model_paths와 system_prompt_dir을 인자로 전달
2. **UI 생성**: Gradio 인터페이스 생성 시작

### **2. ModelBenchmark 클래스 인스턴스 생성**
```python
benchmark = ModelBenchmark(model_paths, system_prompt_dir=system_prompt_dir)
```

**실행 과정:**
1. **클래스 초기화**: ModelBenchmark 클래스의 __init__ 메서드 실행
2. **모델 경로 저장**: model_paths 딕셔너리를 인스턴스 변수로 저장
3. **시스템 프롬프트 로드**: load_system_prompts() 메서드 호출
4. **서버 매니저 초기화**: LlamaServerManager 인스턴스 생성

### **3. 시스템 프롬프트 로드 과정**
```python
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
```

**실행 과정:**
1. **기본 프롬프트 로드**: default_system_prompt.txt 파일에서 기본 프롬프트 로드
2. **모델별 프롬프트 설정**: 모든 모델에 기본 프롬프트 설정
3. **접두사별 프롬프트 로드**: gpt_, gemini_ 등 접두사별 특화 프롬프트 로드
4. **모델별 특화 프롬프트 로드**: 각 모델명별 특화 프롬프트 로드

### **4. Gradio Blocks 인터페이스 생성**
```python
with gr.Blocks(title="sLLM 벤치마크", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# sLLM 모델 벤치마크\n**주의**: 이 도구는 단일 사용자용입니다...")
```

**실행 과정:**
1. **Blocks 컨텍스트 생성**: gr.Blocks로 메인 인터페이스 컨테이너 생성
2. **제목 설정**: "sLLM 벤치마크" 제목 설정
3. **테마 적용**: Soft 테마 적용
4. **마크다운 헤더**: 메인 제목과 주의사항 표시

---

## 🎯 사용자 인터랙션 처리 과정

### **1. 탭별 UI 구성**
```python
with gr.Tab("벤치마크 실행 및 시각화"):
    # 벤치마크 탭 UI 구성
    
with gr.Tab("질의 생성 벤치마크"):
    # 질의 생성 탭 UI 구성
    
with gr.Tab("QA 챗봇"):
    # QA 챗봇 탭 UI 구성
    
with gr.Tab("지식베이스 생성"):
    # 지식베이스 생성 탭 UI 구성
```

**실행 과정:**
1. **탭 생성**: 각 기능별로 탭 생성
2. **UI 컴포넌트 배치**: 각 탭 내부에 필요한 UI 컴포넌트들 배치
3. **이벤트 핸들러 연결**: 버튼 클릭, 입력 변경 등의 이벤트 핸들러 연결

### **2. 이벤트 핸들러 등록**
```python
run_button.click(
    run_benchmark_task, 
    [benchmark_models, use_sys_prompt, ...], 
    [progress_output, results_json]
)
```

**실행 과정:**
1. **이벤트 바인딩**: 버튼 클릭 이벤트를 함수에 바인딩
2. **입력 파라미터 정의**: 함수에 전달할 입력 파라미터들 정의
3. **출력 컴포넌트 정의**: 함수 결과를 표시할 출력 컴포넌트들 정의

---

## 🔄 각 탭별 실행 흐름

### **1. 벤치마크 실행 및 시각화 탭**

#### **UI 구성 과정**
```python
with gr.Tab("벤치마크 실행 및 시각화"):
    with gr.Row():
        with gr.Column(scale=1):
            # 설정 컬럼
            benchmark_models = gr.CheckboxGroup(choices=list(model_paths.keys()))
            query_file = gr.File(label="쿼리 JSON 파일")
            # ... 기타 설정 컴포넌트들
        with gr.Column(scale=2):
            # 결과 컬럼
            progress_output = gr.Textbox(label="진행 상황")
            results_json = gr.JSON(label="결과 JSON")
```

#### **실행 버튼 클릭 시 처리 과정**
```python
def run_benchmark_task(models, use_sys, use_simplification, ...):
    # 1. 입력 검증
    prompts = [p.strip() for p in prompts_str.split('\n') if p.strip()]
    if not prompts: 
        raise gr.Error("No prompts specified.")
    
    # 2. 복합질의 단순화 (선택적)
    if use_simplification and QUERY_SIMPLIFICATION_AVAILABLE:
        simplified_prompts = []
        for prompt in prompts:
            simplified = simplify_sentence(prompt)
            simplified_prompts.extend(simplified)
        prompts = simplified_prompts
    
    # 3. 모델별 실행
    for model_idx, model in enumerate(models):
        for prompt_idx, prompt in enumerate(prompts):
            if use_evaluation:
                result = generate_with_retry(benchmark, model, prompt, ...)
            else:
                result = benchmark.generate(model, prompt, ...)
            results_data["detailed"][model].append(result)
    
    # 4. 결과 반환
    return f"벤치마크 완료: 총 {total_tasks}개 태스크 실행", results_data
```

### **2. QA 챗봇 탭**

#### **UI 구성 과정**
```python
with gr.Tab("QA 챗봇"):
    with gr.Row():
        with gr.Column(scale=1):
            # 설정 컬럼
            chatbot_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()))
            use_agent_dialogue = gr.Checkbox(value=True, label="Agent 대화 모드 사용")
            # ... 기타 설정
        with gr.Column(scale=2):
            # 대화 컬럼
            chatbot = gr.Chatbot(label="챗봇", height=800)
            chatbot_input = gr.Textbox(lines=2, label="메시지 입력")
```

#### **메시지 전송 시 처리 과정**
```python
def send_message(message, history, model, system_prompt, ...):
    # 1. Agent 대화 모드 (선택적)
    if use_agent_dialogue and agent_model:
        agent_result = benchmark.generate(agent_model, agent_prompt, ...)
        if agent_response.startswith("구조화완료:"):
            # 정보가 충분한 경우
            user_query_for_neo = f"사용자 상황: {message}\n구조화된 정보: {structured_info}"
        elif agent_response.startswith("추가질문:"):
            # 정보가 부족한 경우
            return history, "", f"Agent 추가 질문 완료"
    
    # 2. RAG 검색
    if current_kb_file and CHROMA_RAG_AVAILABLE:
        retrieved_kb = rag_engine.query(user_query_for_neo, top_k=3)
        rag_context = "\n".join(retrieved_kb)
    
    # 3. NEO Query 변환
    neo_query_prompt = system_prompt + "\n\n아래 KB 내용을 참고하여..." + rag_context
    neo_query_result = benchmark.generate(model, neo_query_prompt, ...)
    neo_query = neo_query_result['output'].strip()
    
    # 4. NEO Engine 실행
    if NEO_ENGINE_AVAILABLE and neo_executor is not None:
        result_code, neo_response = neo_executor.execute_query(neo_query)
        neo_result = neo_response.strip()
    
    # 5. 최종 응답 생성
    if use_evaluation:
        result = generate_with_retry(benchmark, model, message, ...)
    else:
        result = benchmark.generate(model, message, ...)
    
    response = result['output']
    history.append((message, response))
    return history, "", status_msg
```

### **3. 지식베이스 생성 탭**

#### **UI 구성 과정**
```python
with gr.Tab("지식베이스 생성"):
    with gr.Row():
        with gr.Column(scale=1):
            # 설정 컬럼
            kb_gen_model_dropdown = gr.Dropdown(choices=list(model_paths.keys()))
            kb_gen_input_type = gr.Radio(choices=["텍스트 입력", "파일 업로드", "URL 입력"])
            # ... 기타 설정
        with gr.Column(scale=2):
            # 결과 컬럼
            kb_gen_output = gr.Textbox(lines=20, label="생성된 지식베이스")
```

#### **생성 버튼 클릭 시 처리 과정**
```python
def generate_knowledge_base(model, system_prompt, input_type, text_input, file_input, url_input, ...):
    # 1. 입력 데이터 추출
    if input_type == "텍스트 입력":
        input_text = text_input.strip()
    elif input_type == "파일 업로드":
        if file_ext == '.pdf':
            pdf_chunks = extract_pdf_chunks_from_file(file_input)
        else:
            input_text = extract_text_from_file(file_input)
    elif input_type == "URL 입력":
        input_text = extract_text_from_url(url_input)
    
    # 2. LLM 기반 KB 변환
    format_instructions = {
        "NEO 형식 (.kb)": "NEO 엔진에서 사용할 수 있는 S-식 형태의 지식베이스를 생성하세요...",
        "자연어 형식 (.nkb)": "자연어로 된 지식베이스를 생성하세요...",
        "JSON 형식 (.json)": "JSON 형태의 구조화된 지식베이스를 생성하세요..."
    }
    
    final_prompt = f"{system_prompt}\n\n=== 입력 데이터 ===\n{input_text}\n\n=== 출력 형식 지침 ===\n{format_instruction}"
    result = benchmark.generate(model, final_prompt, ...)
    
    # 3. 결과 저장
    file_extensions = {"NEO 형식 (.kb)": ".kb", "자연어 형식 (.nkb)": ".nkb", "JSON 형식 (.json)": ".json"}
    file_ext = file_extensions.get(output_format, ".kb")
    file_path = os.path.join(kb_dir, f"{filename}{file_ext}")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(generated_kb)
    
    return f"지식베이스 생성 완료: {file_path}", generated_kb
```

---

## 🔧 핵심 클래스 및 함수 실행 과정

### **1. ModelBenchmark.generate() 메서드**

#### **실행 과정**
```python
def generate(self, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float, use_system_prompt: bool, openai_api_key: Optional[str], gemini_api_key: Optional[str], port: int, gpu_layers: int) -> Dict[str, Any]:
    start_time = time.time()
    system_prompt = self.system_prompts.get(model_name, "") if use_system_prompt else ""
    
    try:
        # 1. 모델 타입 확인
        model_type = self.model_paths.get(model_name, "")
        
        # 2. OpenAI 모델 처리
        if model_name.startswith("gpt-"):
            if not openai_api_key: 
                raise ValueError("OpenAI API key is required.")
            headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            messages.append({"role": "user", "content": prompt})
            json_data = {"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            output = data['choices'][0]['message']['content']
            return {
                "output": output, 
                "elapsed_time": time.time() - start_time,
                "tokens_generated": data.get('usage', {}).get('completion_tokens', 0),
                "tokens_prompt": data.get('usage', {}).get('prompt_tokens', 0),
                "tokens_total": data.get('usage', {}).get('total_tokens', 0),
                "system_prompt": system_prompt, 
                "prompt": prompt, 
                "model": model_name
            }
        
        # 3. Gemini 모델 처리
        elif model_name.startswith("gemini"):
            if not gemini_api_key: 
                raise ValueError("Gemini API key is required.")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"
            full_prompt = f"{system_prompt}\n\n{prompt}".strip()
            json_data = {"contents": [{"parts": [{"text": full_prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature, "topP": top_p}}
            response = requests.post(url, json=json_data)
            response.raise_for_status()
            data = response.json()
            output = data['candidates'][0]['content']['parts'][0]['text']
            return {
                "output": output, 
                "elapsed_time": time.time() - start_time, 
                "system_prompt": system_prompt,
                "prompt": prompt, 
                "model": model_name
            }
        
        # 4. 로컬 모델 처리
        else:
            model_path = self.model_paths.get(model_name)
            if not model_path:
                raise ValueError(f"model_paths에 '{model_name}'가 없습니다.")
            if not self.server_manager.start(model_path, self.context_size, port, gpu_layers):
                raise ConnectionError(f"Failed to start local server for {model_name}.")
            headers = {"Content-Type": "application/json"}
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            messages.append({"role": "user", "content": prompt})
            json_data = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
            response = requests.post(f"http://{SERVER_HOST}:{port}/v1/chat/completions", headers=headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            output = data['choices'][0]['message']['content']
            return {
                "output": output, 
                "elapsed_time": time.time() - start_time,
                "tokens_generated": data.get('usage', {}).get('completion_tokens', 0),
                "tokens_prompt": data.get('usage', {}).get('prompt_tokens', 0),
                "tokens_total": data.get('usage', {}).get('total_tokens', 0),
                "system_prompt": system_prompt, 
                "prompt": prompt, 
                "model": model_name
            }
            
    except Exception as e:
        return {"error": str(e), "prompt": prompt, "model": model_name, "system_prompt": system_prompt, "output": output if output is not None else ""}
```

### **2. LlamaServerManager.start() 메서드**

#### **실행 과정**
```python
def start(self, model_path: str, ctx_size: int, port: int, gpu_layers: int) -> bool:
    self.stop()  # 기존 서버 중지
    
    # 1. 서버 명령어 구성
    cmd = [
        self.server_path, 
        "-m", model_path, 
        "--ctx-size", str(ctx_size), 
        "--host", self.host, 
        "--port", str(port), 
        "-ngl", str(gpu_layers)
    ]
    
    # 2. 서버 프로세스 시작
    try:
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        self.current_model_info = {"model_path": model_path, "port": port, "gpu_layers": gpu_layers}
        
        # 3. 서버 준비 대기
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
```

### **3. generate_with_retry() 함수**

#### **실행 과정**
```python
def generate_with_retry(benchmark, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float, use_system_prompt: bool, openai_api_key: Optional[str], gemini_api_key: Optional[str], port: int, gpu_layers: int, evaluation_model: str, max_retries: int = 3, system_prompt_dir: str = "./templates") -> Dict[str, Any]:
    attempts = 0
    evaluation_results = []
    start_time = time.time()
    max_total_time = 300  # 최대 5분 제한
    
    # 원본 시스템 프롬프트 저장
    original_system_prompt = benchmark.get_system_prompt(model_name)
    
    while attempts < max_retries:
        attempts += 1
        
        # 1. 모델 생성
        result = benchmark.generate(model_name, prompt, max_tokens, temperature, top_p, use_system_prompt, openai_api_key, gemini_api_key, port, gpu_layers)
        
        if 'error' in result:
            return result
        
        output = result['output']
        
        # 2. 출력 평가
        evaluation = evaluate_output(prompt, output, evaluation_model, port, gpu_layers, openai_api_key, gemini_api_key, system_prompt_dir)
        evaluation_results.append(evaluation)
        
        # 3. 평가 결과에 따른 분기
        if evaluation['pass']:
            # 통과 시 결과 반환
            result['attempts'] = attempts
            result['evaluation_results'] = evaluation_results
            benchmark.set_system_prompt(model_name, original_system_prompt)
            return result
        else:
            # 불통과 시 재생성 (최대 재시도 횟수까지)
            if attempts < max_retries:
                time.sleep(2)  # 재생성 전 잠시 대기
    
    # 최대 시도 횟수 도달 시 마지막 결과 반환
    result['attempts'] = attempts
    result['evaluation_results'] = evaluation_results
    result['warning'] = f"최대 재시도 횟수({max_retries})에 도달했습니다."
    benchmark.set_system_prompt(model_name, original_system_prompt)
    return result
```

---

## 🛑 종료 및 정리 과정

### **1. Gradio 인터페이스 종료**
```python
demo.close(benchmark.server_manager.stop)
```

**실행 과정:**
1. **종료 핸들러 등록**: demo.close()에 서버 중지 함수 등록
2. **사용자 종료 시**: 서버 프로세스 자동 중지

### **2. LlamaServerManager.stop() 메서드**
```python
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
```

**실행 과정:**
1. **프로세스 확인**: 실행 중인 서버 프로세스가 있는지 확인
2. **정상 종료 시도**: terminate()로 정상 종료 시도
3. **강제 종료**: 5초 후에도 종료되지 않으면 kill()로 강제 종료
4. **정리**: 프로세스 참조와 모델 정보 초기화

---

## 📊 실행 과정 요약

### **전체 실행 흐름**
```mermaid
graph TD
    A[python run.py 실행] --> B[명령행 인자 파싱]
    B --> C[필수 라이브러리 임포트]
    C --> D[NEO Engine/Chroma RAG 임포트]
    D --> E[전역 변수 초기화]
    E --> F[GGUF 모델 파일 탐색]
    F --> G[ModelBenchmark 인스턴스 생성]
    G --> H[시스템 프롬프트 로드]
    H --> I[Gradio UI 생성]
    I --> J[이벤트 핸들러 등록]
    J --> K[demo.launch() 실행]
    K --> L[웹 브라우저에서 인터페이스 접근]
    L --> M[사용자 인터랙션 처리]
    M --> N[프로그램 종료 시 서버 정리]
```

### **주요 실행 단계별 시간**
| 단계 | 예상 시간 | 설명 |
|------|-----------|------|
| **초기화** | 1-2초 | 라이브러리 임포트, 변수 초기화 |
| **모델 탐색** | 2-5초 | GGUF 파일 스캔, 경로 설정 |
| **UI 생성** | 3-5초 | Gradio 인터페이스 구성 |
| **서버 시작** | 10-30초 | 로컬 모델 서버 시작 (첫 실행 시) |
| **사용자 작업** | 가변 | 사용자의 실제 작업 시간 |

### **메모리 사용량**
| 구성요소 | 메모리 사용량 | 설명 |
|----------|---------------|------|
| **Python 프로세스** | 100-200MB | 기본 Python 런타임 |
| **Gradio 인터페이스** | 50-100MB | 웹 인터페이스 |
| **로컬 모델 서버** | 2-8GB | 모델 크기에 따라 변동 |
| **Chroma RAG** | 100-500MB | 벡터 데이터베이스 |
| **NEO Engine** | 50-200MB | 논리적 추론 엔진 |

---

*이 문서는 NEO 벤치마크 시스템의 run.py 실행 과정을 상세히 설명합니다. 각 단계는 실제 코드 실행 순서를 따라 작성되었으며, 디버깅과 성능 최적화에 도움이 될 것입니다.*

