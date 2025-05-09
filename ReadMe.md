# NEO_LLM 프로젝트

이 프로젝트는 **NEO 언어와 소형 대규모 언어 모델(sLLM)**을 통합하는 것을 목표로 합니다. 프로젝트의 핵심은 **NEO 언어를 활용한 지식 베이스(KB) 구축과 질의 처리**입니다.

<br>

---
<br>

## 프로젝트 개요
본 프로젝트는 **NEO_LLM 시스템**을 기반으로 하여 **자연어 질의(NLQ)를 NEO 언어로 변환**하고, 이를 통해 더 나은 품질의 응답과 정보 검색을 수행하는 것을 목표로 합니다.

핵심 실행 파일인 `run.py`는 프로젝트의 중심 역할을 하며, 주요 모듈들을 호출하여 다음과 같은 작업을 수행합니다.

1. **사용자가 입력한 문서를 LLM을 통해 단순 문장으로 변환**
2. **Construction Grammar 처리 후, 벡터 저장소(Vectorstore) 생성**
3. **NEO 언어로 변환하여 지식 베이스(KB) 구축**
4. **사용자의 질의를 변환하고 KB를 기반으로 논리 추론 수행**
5. **추론 결과를 LLM에 반영하여 최적화된 응답 생성**

<br>

---
<br>

## 프로젝트 구조

![Flowchart](https://github.com/NangManCatDv/MOADATA/blob/main/flowchart/flowchart.drawio.png)


### 실행 파일
- **[run.py](https://github.com/NangManCatDv/MOADATA/blob/main/run.py)**:
  - 프로젝트의 핵심 실행 파일
  - 문서 처리, KB 구축, 질의 변환 및 응답 생성 전 과정을 담당

### 주요 모듈
- **[ollama_preprocessor.py](https://github.com/NangManCatDv/MOADATA/blob/main/ollama_preprocessor.py)**:
  - LLM을 활용하여 복잡한 문장을 단순 문장으로 변환
  - 구문 분석(Construction Grammar) 전처리 수행

- **[data_preprocessor.py](https://github.com/NangManCatDv/MOADATA/blob/main/data_preprocessor.py)**:
  - `ollama_preprocessor.py`에서 생성된 단순 문장에 대해 Construction Grammar 처리

- **[vectorstore.py](https://github.com/NangManCatDv/MOADATA/blob/main/vectorstore.py)**:
  - `data_preprocessor.py`의 결과물을 기반으로 벡터 저장소(Vectorstore) 생성

- **[vectorstore_to_NEO.py](https://github.com/NangManCatDv/MOADATA/blob/main/vectorstore_to_NEO.py)**:
  - 벡터 저장소를 NEO 언어 기반 지식 베이스(KB)로 변환
  - LLM의 시스템 프롬프트를 활용하여 KB 구축

- **[query_to_NEO.py](https://github.com/NangManCatDv/MOADATA/blob/main/query_to_NEO.py)**:
  - 사용자의 질의를 변환하고 KB를 참조하여 질의어(Query) 생성

- **[NEO_executor.py](https://github.com/NangManCatDv/MOADATA/blob/main/NEO_executor.py)**:
  - 생성된 질의어를 NEO엔진에서 실행하여 결과를 도출

- **[response.py](https://github.com/NangManCatDv/MOADATA/blob/main/response.py)**:
  - KB에서 도출된 결과를 LLM에 삽입하여 응답 생성
  - 보다 최적화된 답변을 제공하도록 개선

<br>

---
<br>

## 개발 진행 상황
### [초기 환경 설정 & 기본 구조 구축](https://github.com/NangManCatDv/MOADATA/milestone/2)

### [코드 개선 및 응답 정확도 향상](https://github.com/NangManCatDv/MOADATA/milestone/3)

### [테스트 및 안정화 그리고 최종 배포](https://github.com/NangManCatDv/MOADATA/milestone/5)

<br>

---
<br>

## 참고 문헌
- **[2013-지능정보-추계-한국어의 Construction Grammar구현방안연구-2013-11-29-revised.pdf](https://drive.google.com/file/d/1DX9QclWba1iZJDF5vxcv_SjsvXyiq0C4/view?usp=sharing)**

<br>

---
<br>

## 작업환경 💻

- OS: Windows 10 Home [64비트]
- Host: Micro-Star International Co., Ltd. MS-7C94
- Kernel: 10.0.19045.0
- Motherboard: Micro-Star International Co., Ltd. MAG B550M MORTAR (MS-7C94)
- CPU: AMD Ryzen 5 5600X 6-Core Processor @ 3.7GHz
- GPU: NVIDIA GeForce RTX 3060 Ti
- Memory: 31.93 GiB
- Python 3.12.5

<br>

---
<br>

## 저장소 정보
- **프로젝트 책임자:** *NangManCatDev*

이 프로젝트는 **NEO 기반 논리 추론과 LLM의 자연어 처리 능력을 결합**하여 **설명 가능 AI 및 강력한 지식 기반 시스템을 구축**하는 것을 목표로 합니다.
