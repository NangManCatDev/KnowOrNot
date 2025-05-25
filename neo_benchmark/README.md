# NEO Knowledge Base 변환 벤치마크

이 프로젝트는 다양한 LLM(Large Language Model) 모델을 사용하여 자연어 문장을 NEO Knowledge Base 형식으로 변환하는 성능을 벤치마크하는 도구입니다. 여러 모델을 동시에 테스트하고 결과를 비교할 수 있습니다.

## 주요 기능

- 자연어 텍스트를 NEO Knowledge Base 형식으로 변환
- 여러 LLM 모델의 변환 성능 비교
- 변환 시간, 토큰 생성 속도 등 다양한 성능 지표 측정
- 벤치마크 결과의 시각화

## 설치 방법

### 선행 요구 사항

- Python 3.8 이상
- llama.cpp 서버 (빌드된 llama-server.exe)
- GGUF 형식의 LLM 모델 파일

### 설치 단계

1. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/neo-knowledge-base-benchmark.git
   cd neo-knowledge-base-benchmark
   ```

2. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

3. `llama.cpp` 설치 (이미 설치되어 있다면 이 단계 생략)
   - https://github.com/ggerganov/llama.cpp 저장소를 클론
   - 지침에 따라 빌드하여 `llama-server.exe` 생성

## 사용 방법

### 기본 사용법

```bash
python neo_benchmark/run_benchmark.py --models_dir "./models" --external_models_dir "C:/path/to/your/models" --queries_file "sample_queries.json" --llama_server_path "C:/path/to/llama-server.exe"
```

### 주요 명령행 옵션

- `--models_dir`: 모델 파일이 저장된 디렉토리 경로
- `--external_models_dir`: 외부 모델 파일이 저장된 디렉토리 경로
- `--queries_file`: 벤치마크에 사용할 질의가 저장된 JSON 파일 경로
- `--output_dir`: 벤치마크 결과를 저장할 디렉토리 경로 (기본값: "./benchmark_results")
- `--context_size`: 모델의 최대 컨텍스트 길이 (기본값: 2048)
- `--system_prompt_dir`: 시스템 프롬프트 템플릿이 저장된 디렉토리 경로
- `--no_system_prompt`: 시스템 프롬프트를 사용하지 않음
- `--llama_server_path`: llama-server 실행 파일 경로
- `--host`: 서버 호스트 주소 (기본값: "127.0.0.1")
- `--port`: 서버 포트 번호 (기본값: 8080)

### 모델 인식

벤치마크 도구는 지정된 디렉토리에서 자동으로 GGUF 형식(.gguf)의 모델 파일을 찾아 사용합니다. 다음 디렉토리에서 모델을 찾습니다:

1. `--models_dir` 옵션으로 지정한 디렉토리
2. `--external_models_dir` 옵션으로 지정한 디렉토리

모델 파일명이 모델 이름으로 사용됩니다 (예: "llama2-13b-chat.gguf" → "llama2-13b-chat").

### 시스템 프롬프트 관리

시스템 프롬프트는 자연어를 NEO Knowledge Base 형식으로 변환하는 방법을 모델에 지시하는 데 사용됩니다. 모델별 커스텀 시스템 프롬프트를 지정하려면:

1. 기본 시스템 프롬프트: `templates/default_system_prompt.txt`
2. 모델군별 시스템 프롬프트: `templates/{model_prefix}_system_prompt.txt` (예: "llama_system_prompt.txt")
3. 특정 모델 시스템 프롬프트: `templates/{model_name}_system_prompt.txt` (예: "llama2-13b-chat_system_prompt.txt")

우선순위는 3 > 2 > 1 순입니다.

## 결과 해석

벤치마크 결과는 다음 파일에 저장됩니다:

- `knowledge_base_results_{timestamp}.csv`: 전체 결과 데이터 (시스템 프롬프트는 파일 상단에 한 번만 표시)
- `kb_performance_summary_{timestamp}.csv`: 성능 요약 데이터
- `kb_processing_time_{timestamp}.png`: 모델별 처리 시간 그래프
- `kb_tokens_per_second_{timestamp}.png`: 모델별 토큰 생성 속도 그래프
- `kb_tokens_generated_{timestamp}.png`: 모델별 생성 토큰 수 그래프
- `system_prompts_{timestamp}.json`: 사용된 시스템 프롬프트

## 질의 형식

`sample_queries.json` 파일에는 각 카테고리별로 테스트할 자연어 문장들이 포함되어 있습니다. 다음과 같은 형식으로 구성됩니다:

```json
{
  "카테고리1": [
    "자연어 문장 1",
    "자연어 문장 2"
  ],
  "카테고리2": [
    "자연어 문장 3",
    "자연어 문장 4"
  ]
}
```

자신만의 질의 파일을 작성하여 사용할 수 있습니다.

## 주의사항

1. 고성능 모델(13B 이상)을 사용할 경우 충분한 VRAM이 필요합니다.
2. 경로에 공백이나 특수문자가 포함된 경우 오류가 발생할 수 있습니다.
3. 결과는 하드웨어 성능, 모델 크기, 자연어 문장의 복잡성 등 여러 요인에 따라 달라질 수 있습니다.

## 라이선스

MIT 라이선스 