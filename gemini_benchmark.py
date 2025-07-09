import os
import time
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import functools 

import gradio as gr
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class GeminiModelBenchmark:
    def __init__(self, gemini_model_names: List[str], system_prompt_dir: str = "./templates"):
        self.model_names = gemini_model_names
        self.system_prompt_dir = Path(system_prompt_dir)
        self.system_prompts = {}
        self.gemini_api_key: Optional[str] = None
        self.load_system_prompts()
        
    def load_system_prompts(self) -> None:
        default_prompt_path = self.system_prompt_dir / "default_system_prompt.txt"
        if default_prompt_path.exists():
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                default_prompt = f.read().strip()
        else:
            default_prompt = "You are a helpful AI assistant. Provide accurate and helpful answers."
            
        for model_name in self.model_names:
            self.system_prompts[model_name] = default_prompt
            
        for model_name in self.model_names:
            prompt_path = self.system_prompt_dir / f"{model_name}_system_prompt.txt"
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    self.system_prompts[model_name] = f.read().strip()
            else:
                generic_gemini_prompt_path = self.system_prompt_dir / "gemini_system_prompt.txt"
                if generic_gemini_prompt_path.exists():
                     with open(generic_gemini_prompt_path, 'r', encoding='utf-8') as f:
                        self.system_prompts[model_name] = f.read().strip()

    def get_available_templates(self) -> List[str]:
        templates = []
        if self.system_prompt_dir.exists():
            for file in self.system_prompt_dir.glob("*_system_prompt.txt"):
                templates.append(file.name)
        return sorted(templates)
    
    def load_template_content(self, template_name: str) -> str:
        if not template_name:
            return ""
        template_path = self.system_prompt_dir / template_name
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""
    
    def set_system_prompt(self, model_name: str, system_prompt: str) -> None:
        if model_name not in self.model_names:
            print(f"[WARN] Model {model_name} not configured. Ignoring set_system_prompt.")
            return
        self.system_prompts[model_name] = system_prompt
        
    def get_system_prompt(self, model_name: str) -> str:
        return self.system_prompts.get(model_name, "You are a helpful AI assistant.")
    
    def save_system_prompt(self, model_name: str, filename: Optional[str] = None) -> None:
        if model_name not in self.model_names:
            print(f"[WARN] Model {model_name} not configured. Ignoring save_system_prompt.")
            return
        if filename is None:
            filename = f"{model_name}_system_prompt.txt"
        prompt_path = self.system_prompt_dir / filename
        os.makedirs(self.system_prompt_dir, exist_ok=True)
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(self.system_prompts[model_name])

    def set_gemini_api_key(self, gemini_key: Optional[str]):
        self.gemini_api_key = gemini_key
        print(f"[DEBUG] Instance Gemini API Key Set: {'SET' if gemini_key else 'NOT SET'}")

    def generate(self, model_name: str, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.7, top_p: float = 0.95, 
                 use_system_prompt: bool = True) -> Dict[str, Any]:
        if not model_name.startswith("gemini"):
            raise Exception(f"Model {model_name} is not a Gemini model.")
        if not self.gemini_api_key:
            raise Exception("Gemini API 키가 필요합니다.")

        start_time = time.time()
        full_prompt_parts = []
        current_system_prompt = ""
        if use_system_prompt and model_name in self.system_prompts:
            current_system_prompt = self.system_prompts[model_name].strip()
            if current_system_prompt:
                 full_prompt_parts.append({"text": current_system_prompt})
        full_prompt_parts.append({"text": prompt})

        # Use v1beta endpoint as per original structure, ensure model_name is compatible
        # The error indicates "gemini-1.0-pro" is problematic with v1beta for generateContent
        api_version = "v1beta"
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.gemini_api_key}
        
        contents_payload = [{"parts": full_prompt_parts}]
        data = {
            "contents": contents_payload,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
        }

        response = requests.post(url, headers=headers, params=params, json=data)
        
        if response.status_code != 200:
            # Include the response text in the exception for more details
            raise Exception(f"Gemini API 오류: {response.status_code} - {response.text}") 
            
        response_data = response.json()
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        try:
            if not response_data.get('candidates') or \
               not response_data['candidates'][0].get('content') or \
               not response_data['candidates'][0]['content'].get('parts'):
                model_output = f"No content in response: {response_data}"
            else:
                model_output = response_data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError, TypeError) as e:
            model_output = f"Error parsing Gemini response: {str(e)}. Full response: {response_data}"

        tokens_generated = int(len(model_output.split()) * 1.3)
        tokens_prompt = int(len(prompt.split()) * 1.3 + (len(current_system_prompt.split()) *1.3 if current_system_prompt else 0) )
        tokens_total = tokens_prompt + tokens_generated
        
        return {
            "model": model_name,
            "prompt": prompt,
            "system_prompt": current_system_prompt if use_system_prompt else "",
            "output": model_output,
            "elapsed_time": elapsed_time,
            "tokens_generated": tokens_generated,
            "tokens_prompt": tokens_prompt,
            "tokens_total": tokens_total
        }

    def benchmark(self, prompts: List[str], model_names_to_run: Optional[List[str]] = None, 
                 use_system_prompt: bool = True) -> Dict[str, Any]:
        if model_names_to_run is None:
            model_names_to_run = self.model_names
        else:
            model_names_to_run = [m for m in model_names_to_run if m in self.model_names]

        results = {model_name: [] for model_name in model_names_to_run}
        
        for model_name in model_names_to_run:
            for prompt_item in prompts:
                try:
                    result = self.generate(model_name, prompt_item, use_system_prompt=use_system_prompt)
                    results[model_name].append(result)
                except Exception as e:
                    print(f"[ERROR] Failed to generate for {model_name} with prompt '{prompt_item}': {e}")
                    results[model_name].append({
                        "model": model_name, "prompt": prompt_item,
                        "system_prompt": self.get_system_prompt(model_name) if use_system_prompt else "",
                        "output": f"오류: {str(e)}", "elapsed_time": 0,
                        "tokens_generated": 0, "tokens_prompt": 0, "tokens_total": 0
                    })
        
        summary = {}
        for model_name, model_results in results.items():
            if not model_results:
                summary[model_name] = {"total_time": 0, "avg_time_per_prompt": 0, "total_tokens_generated": 0, "avg_tokens_per_prompt": 0, "tokens_per_second": 0}
                continue
            total_time = sum(r["elapsed_time"] for r in model_results if "elapsed_time" in r)
            total_tokens_generated = sum(r["tokens_generated"] for r in model_results if "tokens_generated" in r)
            summary[model_name] = {
                "total_time": total_time,
                "avg_time_per_prompt": total_time / len(prompts) if prompts else 0,
                "total_tokens_generated": total_tokens_generated,
                "avg_tokens_per_prompt": total_tokens_generated / len(prompts) if prompts else 0,
                "tokens_per_second": total_tokens_generated / total_time if total_time > 0 else 0
            }
        self.results = {"detailed": results, "summary": summary}
        return self.results
    
    def save_results(self, output_file: str) -> None:
        results_dir = Path("gemini_benchmark_results")
        results_dir.mkdir(exist_ok=True)
        if not Path(output_file).parent.name:
             output_path = results_dir / output_file
        else:
             output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"Benchmark results saved to {output_path}")

def create_gemini_benchmark_interface(gemini_model_names: List[str], system_prompt_dir: str = "./templates"):
    benchmark = GeminiModelBenchmark(gemini_model_names, system_prompt_dir=system_prompt_dir)
    available_templates = benchmark.get_available_templates()
    results_dir = Path("gemini_benchmark_results")
    results_dir.mkdir(exist_ok=True)

    with gr.Blocks(title="Gemini 모델 벤치마크", analytics_enabled=False) as demo:
        gr.Markdown("# Gemini 모델 벤치마크")
        
        with gr.Tab("단일 모델 테스트"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Gemini 모델 단일 테스트")
                    model_dropdown = gr.Dropdown(choices=gemini_model_names, label="Gemini 모델 선택", value=gemini_model_names[0] if gemini_model_names else None)
                    system_prompt_template = gr.Dropdown(choices=[""] + available_templates, value="", label="시스템 프롬프트 템플릿", allow_custom_value=False)
                    system_prompt_input = gr.Textbox(lines=4, label="시스템 프롬프트", placeholder="시스템 프롬프트를 입력하세요...")
                    prompt_input = gr.Textbox(lines=5, label="프롬프트", placeholder="테스트할 프롬프트를 입력하세요...")
                    with gr.Row():
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.05, label="Top-p")
                    max_tokens = gr.Slider(minimum=16, maximum=8192, value=1024, step=16, label="최대 출력 토큰 수")
                    run_button = gr.Button("실행")
                    output_text = gr.Textbox(lines=10, label="모델 출력")
            def run_single_model(model_name, sys_prompt_text, prompt_val, max_tokens_val, temp_val, top_p_val):
                try:
                    benchmark.set_system_prompt(model_name, sys_prompt_text)
                    result = benchmark.generate(model_name, prompt_val, max_tokens_val, temp_val, top_p_val, True)
                    return result["output"]
                except Exception as e:
                    return f"오류: {str(e)}"
            run_button.click(run_single_model, inputs=[model_dropdown, system_prompt_input, prompt_input, max_tokens, temperature, top_p], outputs=[output_text])
            def update_system_prompt_from_template_single(template_file_name):
                return benchmark.load_template_content(template_file_name)
            system_prompt_template.change(update_system_prompt_from_template_single, inputs=[system_prompt_template], outputs=[system_prompt_input])

        with gr.Tab("모델 비교 벤치마크"):
            with gr.Row():
                with gr.Column():
                    model_checkboxes = gr.CheckboxGroup(choices=gemini_model_names, label="비교할 Gemini 모델 선택", value=gemini_model_names)
                    benchmark_system_prompt_checkbox = gr.Checkbox(value=True, label="시스템 프롬프트 사용")
                    benchmark_prompts = gr.Textbox(lines=8, label="벤치마크 프롬프트 (한 줄에 하나씩)", placeholder="각 프롬프트는 새 줄로 구분합니다.")
                    compare_button = gr.Button("벤치마크 실행 및 저장")
                with gr.Column():
                    benchmark_output_json = gr.JSON(label="벤치마크 결과 (JSON)")
                    result_status = gr.Textbox(label="상태", interactive=False)
            def run_benchmark_compare_and_save(models_list, use_sys_prompt, prompts_text_val):
                try:
                    prompts_list = [p.strip() for p in prompts_text_val.split('\n') if p.strip()]
                    if not prompts_list: return {}, "프롬프트를 입력해주세요."
                    results = benchmark.benchmark(prompts_list, models_list, use_sys_prompt)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"gemini_benchmark_{timestamp}.json"
                    benchmark.save_results(filename)
                    return results, f"{len(models_list)}개 모델, {len(prompts_list)}개 프롬프트 완료. 저장: {results_dir / filename}"
                except Exception as e:
                    return {}, f"오류: {str(e)}"
            compare_button.click(run_benchmark_compare_and_save, inputs=[model_checkboxes, benchmark_system_prompt_checkbox, benchmark_prompts], outputs=[benchmark_output_json, result_status])

        with gr.Tab("시스템 프롬프트 관리"):
            with gr.Row():
                with gr.Column():
                    manage_model_dropdown = gr.Dropdown(choices=gemini_model_names, label="모델 선택", value=gemini_model_names[0] if gemini_model_names else None)
                    manage_template_dropdown = gr.Dropdown(choices=[""] + available_templates, value="", label="템플릿 선택", allow_custom_value=False)
                    load_template_button = gr.Button("템플릿 로드")
                    manage_system_prompt = gr.Textbox(lines=8, label="시스템 프롬프트", placeholder="시스템 프롬프트를 입력하세요...")
                    with gr.Row():
                        load_prompt_button = gr.Button("모델 현재 프롬프트 로드")
                        save_prompt_button = gr.Button("현재 프롬프트 저장")
                    template_name_to_save = gr.Textbox(label="템플릿 파일명 (저장 시 사용)", placeholder="gemini_custom_system_prompt.txt")
                with gr.Column():
                    system_prompt_status = gr.Textbox(label="상태", interactive=False)
                    available_templates_text = gr.Textbox(label="사용 가능한 템플릿", value="\n".join(available_templates) if available_templates else "템플릿 없음", lines=max(5, len(available_templates)), interactive=False)
            def load_template_content_for_manage(template_file_name):
                content = benchmark.load_template_content(template_file_name)
                status_msg = f"'{template_file_name}' 로드 완료." if content else f"'{template_file_name}' 찾을 수 없거나 내용 없음."
                if not template_file_name: status_msg = "템플릿 선택하세요."
                return content, status_msg
            load_template_button.click(load_template_content_for_manage, inputs=[manage_template_dropdown], outputs=[manage_system_prompt, system_prompt_status])
            def load_current_prompt_for_model(model_name_selected):
                if model_name_selected:
                    return benchmark.get_system_prompt(model_name_selected), f"'{model_name_selected}' 프롬프트 로드됨."
                return "", "모델 선택해주세요."
            load_prompt_button.click(load_current_prompt_for_model, inputs=[manage_model_dropdown], outputs=[manage_system_prompt, system_prompt_status])
            def save_current_prompt_for_model(model_name_selected, system_prompt_text, filename_to_save_as):
                if not model_name_selected: return "모델 선택해주세요.", gr.update()
                benchmark.set_system_prompt(model_name_selected, system_prompt_text)
                new_av_temps = benchmark.get_available_templates()
                if filename_to_save_as:
                    benchmark.save_system_prompt(model_name_selected, filename_to_save_as)
                    new_av_temps = benchmark.get_available_templates()
                    msg = f"'{model_name_selected}' 프롬프트를 '{filename_to_save_as}'로 저장."
                else:
                    benchmark.save_system_prompt(model_name_selected)
                    new_av_temps = benchmark.get_available_templates()
                    msg = f"'{model_name_selected}' 프롬프트를 기본 파일명으로 저장."
                return msg, gr.update(value="\n".join(new_av_temps), choices=[""] + new_av_temps)
            save_prompt_button.click(save_current_prompt_for_model, inputs=[manage_model_dropdown, manage_system_prompt, template_name_to_save], outputs=[system_prompt_status, available_templates_text])

        with gr.Tab("API 키 관리"):
            gemini_key_textbox = gr.Textbox(label="Gemini API Key", type="password", placeholder="Enter your Gemini API Key here")
            save_api_keys_button = gr.Button("저장")
            api_keys_status_textbox = gr.Textbox(label="상태", interactive=False)
            def set_keys_in_benchmark(gemini_k):
                benchmark.set_gemini_api_key(gemini_k)
                return f"Gemini API 키가 {'설정됨' if gemini_k else '설정 안됨'}."
            save_api_keys_button.click(set_keys_in_benchmark, inputs=[gemini_key_textbox], outputs=[api_keys_status_textbox])
        
        with gr.Tab("벤치마크 실행 (스트리밍)"):
            with gr.Row():
                with gr.Column():
                    stream_model_checkboxes = gr.CheckboxGroup(choices=gemini_model_names, label="벤치마크할 Gemini 모델 선택", value=gemini_model_names)
                    stream_system_prompt_checkbox = gr.Checkbox(value=True, label="시스템 프롬프트 사용")
                    stream_prompts_textbox = gr.Textbox(lines=5, label="프롬프트 (한 줄에 하나씩)", placeholder="각 프롬프트는 새 줄로 구분합니다.")
                    run_streaming_benchmark_button = gr.Button("스트리밍 벤치마크 시작")
                with gr.Column():
                    benchmark_progress_text = gr.Textbox(label="진행 상태", interactive=False)
                    results_dataframe = gr.DataFrame(label="결과")
            def run_streaming_benchmark(models_to_run, use_sys_prompt_val, prompts_text_val):
                prompts_list = [p.strip() for p in prompts_text_val.split('\n') if p.strip()]
                if not prompts_list: yield "프롬프트를 입력해주세요.", pd.DataFrame(); return
                all_results_data = []
                total_tasks = len(models_to_run) * len(prompts_list); completed_tasks = 0
                for model_name_item in models_to_run:
                    for prompt_val_item in prompts_list:
                        completed_tasks += 1
                        progress_message = f"진행: {model_name_item} - 프롬프트 {prompts_list.index(prompt_val_item) + 1}/{len(prompts_list)} ({completed_tasks}/{total_tasks})"
                        try:
                            result = benchmark.generate(model_name_item, prompt_val_item, use_system_prompt=use_sys_prompt_val)
                            result_data = { "모델": model_name_item, "프롬프트": prompt_val_item, "출력": result.get("output", ""),
                                            "시스템 프롬프트": result.get("system_prompt", ""), "처리 시간(초)": result.get("elapsed_time", 0),
                                            "생성된 토큰 수": result.get("tokens_generated", 0), "프롬프트 토큰 수": result.get("tokens_prompt", 0),
                                            "총 토큰 수": result.get("tokens_total", 0) }
                            all_results_data.append(result_data)
                        except Exception as e:
                            all_results_data.append({"모델": model_name_item, "프롬프트": prompt_val_item, "출력": f"오류: {str(e)}",
                                                   "시스템 프롬프트": benchmark.get_system_prompt(model_name_item) if use_sys_prompt_val else "",
                                                   "처리 시간(초)": 0, "생성된 토큰 수": 0, "프롬프트 토큰 수": 0, "총 토큰 수": 0 })
                        yield progress_message, pd.DataFrame(all_results_data)
                final_message = f"벤치마크 완료: 총 {completed_tasks}개 작업 처리."
                yield final_message, pd.DataFrame(all_results_data)
            run_streaming_benchmark_button.click(run_streaming_benchmark, inputs=[stream_model_checkboxes, stream_system_prompt_checkbox, stream_prompts_textbox], outputs=[benchmark_progress_text, results_dataframe])

        with gr.Tab("벤치마크 시각화"):
            gr.Markdown("저장된 Gemini 벤치마크 결과(.json)를 시각화합니다.")
            def get_benchmark_json_files(): return [str(f) for f in results_dir.glob("*.json")]
            with gr.Row():
                with gr.Column(scale=1):
                    benchmark_file_dropdown = gr.Dropdown(label="결과 파일 선택", choices=get_benchmark_json_files(), allow_custom_value=True)
                    upload_benchmark_file = gr.File(label="또는 결과 파일 업로드 (.json)", file_types=[".json"])
                    visualize_button = gr.Button("시각화 실행")
                with gr.Column(scale=3):
                    visualization_status = gr.Textbox(label="시각화 상태", interactive=False)
            time_plot = gr.Plot(label="응답 시간 (낮을수록 좋음)"); tokens_ps_plot = gr.Plot(label="초당 토큰 생성 속도 (높을수록 좋음)"); avg_tokens_plot = gr.Plot(label="평균 생성 토큰 수 (참고용)")
            def visualize_gemini_benchmark_results(file_path_or_obj):
                if not file_path_or_obj: return "파일을 선택/업로드해주세요.", None, None, None
                try:
                    file_path = file_path_or_obj.name if hasattr(file_path_or_obj, 'name') else str(file_path_or_obj)
                    if not Path(file_path).exists(): return f"파일 없음: {file_path}", None, None, None
                    with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                    if "summary" not in data: return "올바른 형식 파일 아님 (summary 누락).", None, None, None
                    df = pd.DataFrame.from_dict(data["summary"], orient='index')
                    if df.empty: return "요약 데이터 비어있음.", None, None, None
                    plt.figure(figsize=(10, 6)); df['total_time'].sort_values().plot(kind='barh', color='skyblue'); plt.title('모델별 총 응답 시간 (초)'); plt.xlabel('시간 (초)'); p_time = plt.gcf(); plt.close()
                    plt.figure(figsize=(10, 6)); df['tokens_per_second'].sort_values().plot(kind='barh', color='lightgreen'); plt.title('모델별 초당 토큰 생성 속도'); plt.xlabel('Tokens/Second'); p_tps = plt.gcf(); plt.close()
                    avg_tokens_key = 'avg_tokens_per_prompt' if 'avg_tokens_per_prompt' in df.columns else 'total_tokens_generated' 
                    plt.figure(figsize=(10, 6)); df[avg_tokens_key].sort_values().plot(kind='barh', color='salmon'); plt.title('모델별 평균 생성 토큰 수'); plt.xlabel('토큰 수'); p_avg_tokens = plt.gcf(); plt.close()
                    return f"{Path(file_path).name} 시각화 완료.", p_time, p_tps, p_avg_tokens
                except Exception as e: return f"시각화 오류: {str(e)}", None, None, None
            visualize_button.click(visualize_gemini_benchmark_results, inputs=[benchmark_file_dropdown], outputs=[visualization_status, time_plot, tokens_ps_plot, avg_tokens_plot])
            def upload_and_visualize(file_obj):
                if file_obj:
                    status, p1, p2, p3 = visualize_gemini_benchmark_results(file_obj)
                    return status, p1, p2, p3, gr.update(choices=get_benchmark_json_files(), value=file_obj.name)
                return "파일 업로드 필요.", None, None, None, gr.update()
            upload_benchmark_file.upload(upload_and_visualize, inputs=[upload_benchmark_file], outputs=[visualization_status, time_plot, tokens_ps_plot, avg_tokens_plot, benchmark_file_dropdown])
            benchmark_file_dropdown.change(visualize_gemini_benchmark_results, inputs=[benchmark_file_dropdown], outputs=[visualization_status, time_plot, tokens_ps_plot, avg_tokens_plot])
    return demo

def main():
    default_gemini_models = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        # "gemini-1.0-pro", # 주석 처리: v1beta generateContent에서 404 오류 발생 가능성 있음
    ]
    script_dir = Path(__file__).parent.resolve()
    system_prompt_dir_path = script_dir / "templates"
    system_prompt_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"System prompt directory: {system_prompt_dir_path}")

    demo = create_gemini_benchmark_interface(default_gemini_models, system_prompt_dir=str(system_prompt_dir_path))
    try:
        demo.launch(share=False)
    except Exception as e:
        print(f"Gradio 실행 중 오류 발생: {e}")
    finally:
        print("Gemini Benchmark App 종료.")

if __name__ == "__main__":
    main() 