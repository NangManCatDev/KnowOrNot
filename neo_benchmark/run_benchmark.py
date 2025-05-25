#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
import glob
import subprocess
import requests
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# llama.cpp 서버 관련 설정
LLAMA_SERVER_PATH = "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
API_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
SERVER_PROCESS = None

class KnowledgeBaseConverter:
    """자연어를 NEO Knowledge Base 형식으로 변환하는 벤치마크 클래스"""
    def __init__(self, model_paths: dict, context_size: int = 2048, system_prompt_dir: str = "./templates"):
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
        self.results = {}
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
            default_prompt = "당신은 자연어를 NEO Knowledge Base 형식으로 변환하는 전문가입니다. 사용자의 자연어 입력을 정확한 지식 베이스 형식으로 변환해주세요."
            
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
                    
    def get_system_prompt(self, model_name: str) -> str:
        """특정 모델의 시스템 프롬프트를 반환합니다."""
        return self.system_prompts.get(model_name, "당신은 자연어를 NEO Knowledge Base 형식으로 변환하는 전문가입니다. 사용자의 자연어 입력을 정확한 지식 베이스 형식으로 변환해주세요.")

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
                print(f"서버가 이미 실행 중입니다 ({model_name})...")
                return True
        
        # 기존 서버가 실행 중이면 중지
        self.stop_server()
        
        # 서버 경로 확인
        if not os.path.exists(LLAMA_SERVER_PATH):
            print(f"오류: 서버 실행 파일이 존재하지 않습니다: {LLAMA_SERVER_PATH}")
            return False
        
        model_path = self.model_paths[model_name]
        
        # 모델 경로 확인
        if not os.path.exists(model_path):
            print(f"오류: 모델 파일이 존재하지 않습니다: {model_path}")
            return False
        
        # 절대 경로로 변환
        model_path = os.path.abspath(model_path)
        
        # 명령어 구성 변경 (--n-gpu-layers 사용)
        cmd = [
            LLAMA_SERVER_PATH,
            "-m", model_path,
            "--ctx-size", str(self.context_size),
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--parallel", "1",
            "--n-gpu-layers", "100"  # -ngl 대신 --n-gpu-layers 사용
        ]
        
        cmd_str = " ".join(cmd)
        print(f"서버 실행 명령어: {cmd_str}")
        
        try:
            # 서버 프로세스 시작
            print(f"서버 시작 중 ({model_name})...")
            
            # 새로운 방식으로 프로세스 실행
            SERVER_PROCESS = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 현재 실행 중인 모델 이름 저장
            self.current_model = model_name
            
            # 충분한 시작 시간 대기 (13B 모델은 더 오래 걸릴 수 있음)
            print("서버 시작 대기 중... (100초)")
            time.sleep(100)
            
            # 서버가 실행 중인지 확인
            if SERVER_PROCESS.poll() is not None:
                # 서버가 이미 종료됨
                stdout, stderr = SERVER_PROCESS.communicate()
                print(f"서버가 시작 직후 종료되었습니다.")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                self.stop_server()
                return False
            
            # API 연결 테스트
            print("서버 API 연결 확인 중...")
            max_retries = 5
            for i in range(max_retries):
                try:
                    print(f"API 연결 시도 {i+1}/{max_retries}...")
                    response = requests.get(f"{API_BASE_URL}/models", timeout=5)
                    print(f"서버 응답: {response.status_code}")
                    
                    if response.status_code == 200:
                        print(f"서버 API 연결 성공!")
                        return True
                except Exception as e:
                    print(f"연결 시도 {i+1} 실패: {str(e)}")
                
                if i < max_retries - 1:
                    print("재시도 중...")
                    time.sleep(15)
            
            # 여기까지 왔다면 서버 연결 실패
            print("서버 연결 실패, 서버 종료...")
            self.stop_server()
            return False
            
        except Exception as e:
            print(f"서버 시작 오류: {str(e)}")
            self.stop_server()
            return False
    
    def stop_server(self) -> None:
        """실행 중인 llama-server를 중지합니다."""
        global SERVER_PROCESS
        
        if SERVER_PROCESS is not None:
            print("서버 종료 중...")
            try:
                # 서버 프로세스 종료
                SERVER_PROCESS.terminate()
                # 최대 5초 대기
                SERVER_PROCESS.wait(timeout=5)
                print("서버가 정상적으로 종료되었습니다.")
            except subprocess.TimeoutExpired:
                # 종료되지 않으면 강제 종료
                print("서버 강제 종료...")
                SERVER_PROCESS.kill()
                SERVER_PROCESS.wait()
            except Exception as e:
                print(f"서버 종료 중 오류: {str(e)}")
            
            SERVER_PROCESS = None
            self.current_model = None
    
    def generate(self, model_name: str, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.7, top_p: float = 0.95, 
                 use_system_prompt: bool = True) -> dict:
        """
        모델을 사용하여 텍스트를 생성합니다.
        
        Args:
            model_name: 사용할 모델 이름
            prompt: 입력 프롬프트
            max_tokens: 생성할 최대 토큰 수
            temperature: 생성 온도
            top_p: Top-p 샘플링 파라미터
            use_system_prompt: 시스템 프롬프트 사용 여부
            
        Returns:
            생성 결과와 메타데이터를 포함하는 딕셔너리
        """
        # 서버가 현재 실행 중이고 올바른 모델인지 확인
        if self.current_model != model_name or SERVER_PROCESS is None or SERVER_PROCESS.poll() is not None:
            # 서버가 실행되지 않았거나 다른 모델을 사용 중인 경우
            if not self.start_server(model_name):
                raise Exception(f"llama-server 시작 실패: {model_name}")
        
        # 시스템 프롬프트 적용
        if use_system_prompt and model_name in self.system_prompts:
            system_prompt = self.system_prompts[model_name]
        else:
            system_prompt = ""
        
        start_time = time.time()
        
        # API 요청 구성
        api_url = f"{API_BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        
        # 메시지 구성
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
            # API 요청
            print(f"API 요청 전송 중... ({len(prompt)} 글자)")
            response = requests.post(api_url, headers=headers, json=data, timeout=300)
            
            if response.status_code != 200:
                # 응답 디버깅
                error_msg = f"API 요청 오류: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', response.text)}"
                except:
                    error_msg += f" - {response.text}"
                raise Exception(error_msg)
            
            response_data = response.json()
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 응답 파싱
            if 'choices' not in response_data or not response_data['choices']:
                raise Exception(f"API 응답에 'choices' 필드가 없습니다: {response_data}")
                
            if 'message' not in response_data['choices'][0]:
                raise Exception(f"API 응답에 'message' 필드가 없습니다: {response_data['choices'][0]}")
                
            model_output = response_data['choices'][0]['message']['content']
            
            # 토큰 정보
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
            
            return result
            
        except Exception as e:
            # 상세 오류 메시지
            error_msg = f"API 요청 오류: {str(e)}"
            print(error_msg)
            
            # 서버 상태 확인
            if SERVER_PROCESS is not None and SERVER_PROCESS.poll() is not None:
                stdout, stderr = SERVER_PROCESS.communicate()
                print(f"서버 프로세스가 종료되었습니다.")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
            
            raise Exception(error_msg)
    
    def benchmark(self, prompts: list, model_names=None, use_system_prompt: bool = True) -> dict:
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

def load_queries(queries_file):
    """JSON 파일에서 질의를 로드합니다."""
    with open(queries_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_knowledge_base_benchmark(model_paths, queries, output_dir, context_size=2048, 
                 system_prompt_dir="./templates", use_system_prompt=True):
    """자연어를 NEO Knowledge Base 형식으로 변환하는 벤치마크를 실행합니다."""
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 결과 저장할 폴더 생성
    result_dir = os.path.join(output_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    # 모델 벤치마크 초기화
    benchmark = KnowledgeBaseConverter(model_paths, context_size=context_size, 
                              system_prompt_dir=system_prompt_dir)
    
    # 결과 저장을 위한 데이터프레임 초기화
    results_data = []
    
    # 모델별 시스템 프롬프트 저장
    model_system_prompts = {}
    
    try:
        # 모델별로 서버 시작 (각 모델당 한 번만)
        for model_name in model_paths.keys():
            print(f"\n모델: {model_name} 테스트 중...")
            
            # 서버 시작
            if not benchmark.start_server(model_name):
                print(f"경고: {model_name} 모델의 서버를 시작할 수 없습니다. 다음 모델로 넘어갑니다.")
                continue
            
            # 모델의 시스템 프롬프트 저장
            if use_system_prompt:
                model_system_prompts[model_name] = benchmark.get_system_prompt(model_name)
            
            # 모든 카테고리에 대해 벤치마크 실행
            for category, category_queries in queries.items():
                print(f"\n카테고리: {category}")
                print(f"질의 수: {len(category_queries)}")
                
                for i, query in enumerate(tqdm(category_queries, desc=f"{model_name} / {category} 처리 중")):
                    try:
                        # 모델 실행
                        result = benchmark.generate(model_name, query, use_system_prompt=use_system_prompt)
                        
                        # 결과 저장
                        results_data.append({
                            'category': category,
                            'query_id': i,
                            'input_text': query,
                            'model': model_name,
                            'knowledge_base_output': result['output'],
                            'elapsed_time': result['elapsed_time'],
                            'tokens_generated': result['tokens_generated'],
                            'tokens_prompt': result['tokens_prompt'],
                            'tokens_total': result['tokens_total'],
                            'tokens_per_second': result['tokens_generated'] / result['elapsed_time'] if result['elapsed_time'] > 0 else 0
                        })
                    except Exception as e:
                        print(f"오류 발생: {model_name}, 질의 '{query}': {str(e)}")
                        results_data.append({
                            'category': category,
                            'query_id': i,
                            'input_text': query,
                            'model': model_name,
                            'knowledge_base_output': f"오류: {str(e)}",
                            'elapsed_time': -1,
                            'tokens_generated': 0,
                            'tokens_prompt': 0,
                            'tokens_total': 0,
                            'tokens_per_second': 0
                        })
            
            # 이 모델의 모든 카테고리 테스트 완료 후 서버 중지
            benchmark.stop_server()
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"벤치마크 실행 중 오류: {str(e)}")
    finally:
        # 서버 종료 확인
        benchmark.stop_server()
    
    # 결과를 데이터프레임으로 변환
    if results_data:
        df = pd.DataFrame(results_data)
        
        # 시스템 프롬프트 정보 저장 (별도 파일)
        if use_system_prompt:
            system_prompts_path = os.path.join(result_dir, "system_prompts.json")
            with open(system_prompts_path, 'w', encoding='utf-8') as f:
                json.dump(model_system_prompts, f, ensure_ascii=False, indent=2)
            print(f"시스템 프롬프트 정보가 저장되었습니다: {system_prompts_path}")
        
        # CSV 파일로 저장
        csv_path = os.path.join(result_dir, "knowledge_base_results.csv")
        
        # CSV 파일 저장 시 인코딩과 형식 지정
        df.to_csv(csv_path, index=False, encoding='utf-8', quoting=1)  # quoting=1은 모든 필드를 따옴표로 감싸기
        
        print(f"결과가 저장되었습니다: {result_dir}")
        
        # 결과 시각화 및 분석
        visualize_knowledge_base_results(df, result_dir, timestamp)
        
        return df
    else:
        print("벤치마크 결과가 없습니다. 서버 시작 문제로 인해 벤치마크를 실행할 수 없었습니다.")
        return pd.DataFrame()  # 빈 데이터프레임 반환

def visualize_knowledge_base_results(df, output_dir, timestamp):
    """NEO Knowledge Base 변환 벤치마크 결과를 시각화합니다."""
    # 성능 요약 계산
    performance_summary = df.groupby(['model', 'category']).agg({
        'elapsed_time': 'mean',
        'tokens_generated': 'mean',
        'tokens_per_second': 'mean'
    }).reset_index()
    
    # 비교표 생성
    comparison_table = pd.DataFrame()
    comparison_table['Model'] = df['model'].unique()
    
    # 각 카테고리별 정확도 계산
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        accuracy = category_data.groupby('model').apply(
            lambda x: sum(x['knowledge_base_output'].str.contains('오류') == False) / len(x)
        )
        comparison_table[f'{category} Accuracy'] = accuracy
    
    # 평균 처리 시간
    avg_time = df.groupby('model')['elapsed_time'].mean()
    comparison_table['Average Processing Time (s)'] = avg_time
    
    # 토큰 생성 속도
    avg_tokens_per_sec = df.groupby('model')['tokens_per_second'].mean()
    comparison_table['Average Tokens/Second'] = avg_tokens_per_sec
    
    # 비교표 저장
    comparison_path = os.path.join(output_dir, f"kb_comparison_table_{timestamp}.csv")
    comparison_table.to_csv(comparison_path, index=False, encoding='utf-8')
    print(f"비교표가 저장되었습니다: {comparison_path}")
    
    # 1. Average Processing Time by Model and Category
    plt.figure(figsize=(12, 8))
    
    for model in df['model'].unique():
        model_data = performance_summary[performance_summary['model'] == model]
        plt.bar(
            [f"{cat} ({model})" for cat in model_data['category']], 
            model_data['elapsed_time'],
            label=model
        )
    
    plt.title('Average Processing Time by Model and Category')
    plt.xlabel('Category (Model)')
    plt.ylabel('Average Processing Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, f"kb_processing_time_{timestamp}.png"))
    
    # 2. Token Generation Speed by Model and Category
    plt.figure(figsize=(12, 8))
    
    for model in df['model'].unique():
        model_data = performance_summary[performance_summary['model'] == model]
        plt.bar(
            [f"{cat} ({model})" for cat in model_data['category']], 
            model_data['tokens_per_second'],
            label=model
        )
    
    plt.title('Token Generation Speed by Model and Category')
    plt.xlabel('Category (Model)')
    plt.ylabel('Tokens/Second')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, f"kb_tokens_per_second_{timestamp}.png"))
    
    # 3. Average Generated Tokens by Model and Category
    plt.figure(figsize=(12, 8))
    
    for model in df['model'].unique():
        model_data = performance_summary[performance_summary['model'] == model]
        plt.bar(
            [f"{cat} ({model})" for cat in model_data['category']], 
            model_data['tokens_generated'],
            label=model
        )
    
    plt.title('Average Generated Tokens by Model and Category')
    plt.xlabel('Category (Model)')
    plt.ylabel('Number of Generated Tokens')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, f"kb_tokens_generated_{timestamp}.png"))
    
    # 성능 요약 저장
    summary_path = os.path.join(output_dir, f"kb_performance_summary_{timestamp}.csv")
    performance_summary.to_csv(summary_path, index=False, encoding='utf-8')
    print(f"성능 요약이 저장되었습니다: {summary_path}")

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
    parser = argparse.ArgumentParser(description="자연어를 NEO Knowledge Base 형식으로 변환하는 벤치마크")
    parser.add_argument("--models_dir", type=str, default="./models",
                       help="모델 파일이 저장된 디렉토리 경로")
    parser.add_argument("--external_models_dir", type=str, 
                       default="C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads",
                       help="외부 모델 파일이 저장된 디렉토리 경로")
    parser.add_argument("--queries_file", type=str, default="sample_queries.json",
                       help="벤치마크에 사용할 질의가 저장된 JSON 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="벤치마크 결과를 저장할 디렉토리 경로")
    parser.add_argument("--context_size", type=int, default=2048,
                       help="모델의 최대 컨텍스트 길이")
    parser.add_argument("--system_prompt_dir", type=str, 
                       default="C:/Users/1/Desktop/workplace/code/KnowOrNot/neo_benchmark/templates",
                       help="시스템 프롬프트 템플릿이 저장된 디렉토리 경로")
    parser.add_argument("--no_system_prompt", action="store_true",
                       help="시스템 프롬프트를 사용하지 않음")
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
    
    # 현재 디렉토리 기준으로 경로 처리
    current_dir = Path(__file__).parent
    queries_file = current_dir / args.queries_file
    output_dir = current_dir / args.output_dir
    
    # 시스템 프롬프트 디렉토리 경로 설정
    system_prompt_dir = args.system_prompt_dir
    
    # 여러 디렉토리에서 모델 파일 찾기
    model_paths = find_model_files([args.models_dir, args.external_models_dir])
    
    if not model_paths:
        # 예제 모델 경로 (사용자가 수정해야 함)
        model_paths = {
            "llama-2-7b-chat-q4_K_M": "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads/llama-2-7b-chat.Q4_K_M.gguf",
            "mistral-7b-instruct-q4_K_M": "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        }
        print(f"경고: 모델 디렉토리에서 .gguf 파일을 찾을 수 없습니다.")
        print("샘플 모델 경로를 사용합니다. 실제 모델 경로로 수정해주세요.")
    else:
        print(f"발견된 모델 수: {len(model_paths)}")
        for name, path in model_paths.items():
            print(f"  - {name}: {path}")
    
    # 질의 로드
    if not queries_file.exists():
        print(f"오류: 질의 파일을 찾을 수 없습니다: {queries_file}")
        return
    
    queries = load_queries(queries_file)
    
    # 시스템 프롬프트 디렉토리 확인
    if not os.path.exists(system_prompt_dir):
        print(f"경고: 시스템 프롬프트 디렉토리를 찾을 수 없습니다: {system_prompt_dir}")
        print("기본 시스템 프롬프트를 사용합니다.")
    
    # 벤치마크 실행
    print(f"모델 수: {len(model_paths)}")
    print(f"카테고리 수: {len(queries)}")
    print(f"총 질의 수: {sum(len(q) for q in queries.values())}")
    print(f"시스템 프롬프트 사용: {'아니오' if args.no_system_prompt else '예'}")
    print(f"자연어를 NEO Knowledge Base 형식으로 변환하는 벤치마크를 시작합니다...")
    
    start_time = time.time()
    
    try:
        df = run_knowledge_base_benchmark(
            model_paths, 
            queries, 
            output_dir, 
            context_size=args.context_size,
            system_prompt_dir=system_prompt_dir,
            use_system_prompt=not args.no_system_prompt
        )
        end_time = time.time()
        
        print(f"\n벤치마크 완료! 총 소요 시간: {end_time - start_time:.2f}초")
    except KeyboardInterrupt:
        print("\n벤치마크가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n벤치마크 실행 중 오류 발생: {str(e)}")
    finally:
        # 서버 종료 확인
        benchmark = KnowledgeBaseConverter(model_paths)
        benchmark.stop_server()

if __name__ == "__main__":
    main() 