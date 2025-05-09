import re
import logging
from typing import List
import requests
import os

# FIXME: 전처리 시간이 너무 길어서 이것에 대한 해법 혹은 다른 방법론이 필요해보인다.
# FIXME: 현재는 한번 vectorstore에 저장하고 그 이후에는 저장된 vectorstore를 사용하도록 조치를 해두었다.


# Info: 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def simplify_sentence(sentence: str) -> List[str]:
    """
    복잡한 문장을 CxG(Construction Grammar) 기반으로 더 단순한 문장들로 분리합니다.
    
    Args:
        sentence (str): 분리할 복잡한 문장
        
    Returns:
        List[str]: 단순화된 문장들의 리스트
    """
    prompt = f"""당신은 한국어 문장을 단순화하는 전문가입니다.
주어진 문장을 가능한 단순한 형태의 문장들로 분리해주세요.

예시:
입력: "이순신은 조선 시대의 명장이며, 많은 해전에서 승리를 거두었다."
출력:
이순신은 조선의 명장이다
이순신은 해전에서 승리했다

입력: "그는 부하들을 잘 통솔하고 전략이 뛰어났으며, 백성들의 신뢰를 받았다."
출력:
이순신은 부하들을 통솔했다
이순신은 전략가였다
이순신은 백성들의 신뢰를 받았다

규칙:
1. 각 문장은 반드시 주어를 포함해야 함
2. 각 문장은 하나의 핵심 의미만 전달
3. 가능한 짧고 단순하게 작성
4. 한자는 한글로 변환
5. 괄호 안 내용은 제거
6. 수식어구는 최소화

입력: {sentence}
출력:"""

    try:
        # Info: Ollama API 직접 호출
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   "model": "llama3.1:latest",
                                   "prompt": prompt,
                                   "stream": False,
                                   "temperature": 0.1,
                                   "stop": ["입력:", "규칙:", "예시:"]
                               })
        
        if response.status_code == 200:
            result = response.json()['response']
            # Info: 결과를 줄바꿈을 기준으로 분리하고 빈 줄 제거
            simple_sentences = []
            for line in result.split('\n'):
                line = line.strip()
                if line:
                    # Info: 숫자로 시작하는 부분 제거
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    # Info: 괄호 안의 내용 제거
                    line = re.sub(r'\([^)]*\)', '', line)
                    # Info: 한자 제거
                    line = re.sub(r'[一-龥㐀-䶵豈-龎]+', '', line)
                    # Info: 불필요한 기호 제거
                    line = re.sub(r'[-*|]\s*', '', line)
                    # Info: 불필요한 공백 제거
                    line = re.sub(r'\s+', ' ', line)
                    line = line.strip()
                    
                    # Info: 올바른 문장 구조 확인
                    if line and not any(x in line.lower() for x in ["입력:", "출력:", "예시:", "규칙:"]):
                        # Info: 기본 문형 확인 (주어 + 서술어 구조)
                        if ("은" in line or "는" in line) and ("이다" in line or "했다" in line or "되었다" in line or "받았다" in line):
                            if not any(x in line for x in ["아래", "다음", "위", "이렇게", "처럼"]):
                                simple_sentences.append(line)
            
            return simple_sentences
        else:
            logging.error(f"Ollama API 호출 실패: {response.status_code}")
            return [sentence]
            
    except Exception as e:
        logging.error(f"문장 단순화 중 오류 발생: {str(e)}")
        return [sentence]  # Info: 오류 발생 시 원본 문장 반환

def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리합니다.
    
    Args:
        text (str): 분리할 원본 텍스트
        
    Returns:
        List[str]: 분리된 문장들의 리스트
    """
    # Info: 문장 구분을 위한 정규표현식 패턴
    pattern = r'(?<=[.!?])\s+'
    
    # Info: 텍스트를 문장으로 분리
    sentences = re.split(pattern, text)
    
    # Info: 빈 문장 제거 및 공백 처리
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

def save_sentences_to_file(sentences: List[str], output_path: str = "debug/ollama_preprocessed.txt"):
    """
    전처리된 문장들을 텍스트 파일로 저장합니다.
    
    Args:
        sentences (List[str]): 저장할 문장들의 리스트
        output_path (str): 저장할 파일 경로
    """
    try:
        # Info: 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Info: 파일 저장
        with open(output_path, 'w', encoding='utf-8') as file:
            for sentence in sentences:
                file.write(sentence + '\n')
        
        logging.info(f"✅ 전처리된 문장들을 {output_path}에 저장했습니다.")
        
    except Exception as e:
        logging.error(f"❌ 파일 저장 중 오류 발생: {str(e)}")
        raise

def process_document(doc_path: str = "doc/doc.txt", output_path: str = "debug/ollama_preprocessed.txt") -> List[str]:
    """
    문서를 읽고 문장 단위로 처리한 후, 각 문장을 단순화하고 파일로 저장합니다.
    
    Args:
        doc_path (str): 처리할 문서의 경로
        output_path (str): 전처리된 문장들을 저장할 파일 경로
        
    Returns:
        List[str]: 처리된 단순 문장들의 리스트
    """
    try:
        logging.info(f"🔄 문서 읽기 시작: {doc_path}")
        
        # Info: 파일 읽기
        with open(doc_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Info: 문장 분리
        complex_sentences = split_into_sentences(text)
        logging.info(f"✅ 기본 문장 분리 완료: {len(complex_sentences)}개의 문장 추출")
        
        # Info: 각 문장을 단순화
        simple_sentences = []
        for i, sentence in enumerate(complex_sentences, 1):
            logging.info(f"🔄 문장 단순화 진행 중... ({i}/{len(complex_sentences)})")
            simplified = simplify_sentence(sentence)
            simple_sentences.extend(simplified)
        
        logging.info(f"✅ 문서 처리 완료: {len(simple_sentences)}개의 단순 문장 생성")
        
        # Info: 결과를 파일로 저장
        save_sentences_to_file(simple_sentences, output_path)
        
        return simple_sentences
        
    except FileNotFoundError:
        logging.error(f"❌ 파일을 찾을 수 없습니다: {doc_path}")
        raise
    except Exception as e:
        logging.error(f"❌ 문서 처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # Info: 테스트 실행
    try:
        sentences = process_document()
        print(f"\n총 {len(sentences)}개의 단순화된 문장이 추출되었습니다:")
        for i, sentence in enumerate(sentences, 1):
            print(f"{i}. {sentence}")
        print(f"\n전처리된 문장들은 'doc/preprocessed.txt'에 저장되었습니다.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
