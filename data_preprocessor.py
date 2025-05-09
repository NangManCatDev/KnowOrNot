import os
import json
import logging
import re
from konlpy.tag import Komoran
from typing import Dict, List, Optional, Tuple

# FIXME: 현재 Construction Grammer(CxG)는 개선이 필요하다.
# FIXME: 논문을 따라 그대로 NEO엔진에서 처리를 할까 논의중에 있음


# Info: 로깅 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Info: 데이터 파일 및 저장 경로
DATA_DIR = "debug"
RESULT_DIR = "debug"

# Info: 파일 경로 설정
PREPROCESSED_FILE = os.path.join(DATA_DIR, "ollama_preprocessed.txt")
VECTORSTORE_FILE = os.path.join(RESULT_DIR, "data_preprocessed.json")

# Info: 결과 디렉토리 생성
os.makedirs(RESULT_DIR, exist_ok=True)
logging.info(f"✅ 결과 저장 디렉토리 확인/생성: {RESULT_DIR}")

# Info: 형태소 분석기 초기화
logging.info("🔄 형태소 분석기 초기화 중...")
komoran = Komoran()
logging.info("✅ 형태소 분석기 초기화 완료")

# Info: CxG 구문 패턴 정의
class CxGPattern:
    def __init__(self, name: str, pattern: str, roles: List[str]):
        self.name = name
        self.pattern = pattern
        self.roles = roles

CXG_PATTERNS = [
    CxGPattern(
        name="transitive_action",
        pattern=r"(.+)는 (.+)를 (.+)",
        roles=["agent", "object", "verb"]
    ),
    CxGPattern(
        name="identity",
        pattern=r"(.+)는 (.+)이다",
        roles=["subject", "complement"]
    ),
    CxGPattern(
        name="location_action",
        pattern=r"(.+)는 (.+)에서 (.+)",
        roles=["agent", "location", "verb"]
    ),
    CxGPattern(
        name="dative_action",
        pattern=r"(.+)는 (.+)에게 (.+)를 (.+)",
        roles=["agent", "recipient", "object", "verb"]
    ),
    CxGPattern(
        name="instrumental_action",
        pattern=r"(.+)는 (.+)로 (.+)를 (.+)",
        roles=["agent", "instrument", "object", "verb"]
    )
]

def extract_morphological_features(sentence: str) -> Dict[str, List[str]]:
    """
    문장에서 형태소 분석 결과를 추출합니다.
    """
    morphs = komoran.pos(sentence)
    return {
        "morphemes": morphs,
        "nouns": [word for word, pos in morphs if pos.startswith('N')],
        "verbs": [word for word, pos in morphs if pos.startswith('V')],
        "adjectives": [word for word, pos in morphs if pos.startswith('VA')],
        "particles": [word for word, pos in morphs if pos.startswith('J')]
    }

def match_cxg_pattern(sentence: str) -> Optional[Dict]:
    """
    문장에서 CxG 패턴을 찾아 매칭되는 결과를 반환합니다.
    """
    for pattern in CXG_PATTERNS:
        match = re.match(pattern.pattern, sentence)
        if match:
            groups = match.groups()
            result = {
                "construction": pattern.name,
                "roles": {}
            }
            for role, value in zip(pattern.roles, groups):
                result["roles"][role] = value
            return result
    return None

def process_sentence(sentence: str) -> dict:
    """
    문장을 CxG 기반으로 분석합니다.
    """
    # Info: 기본 형태소 분석
    morph_features = extract_morphological_features(sentence)
    
    # Info: CxG 패턴 매칭
    cxg_analysis = match_cxg_pattern(sentence)
    
    # Info: 결과 통합
    result = {
        "original": sentence,
        "morphological_analysis": morph_features,
    }
    
    if cxg_analysis:
        result.update({
            "construction_type": cxg_analysis["construction"],
            "semantic_roles": cxg_analysis["roles"]
        })
    
    return result

def save_to_json(data: List[Dict], output_file: str):
    """전처리된 데이터를 JSON 파일로 저장"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"✅ JSON 파일 저장 완료: {output_file}")

def process_data():
    """전처리 과정 실행 및 결과 저장"""
    logging.info("🔄 데이터 전처리 시작")

    if os.path.exists(VECTORSTORE_FILE):
        logging.info(f"✅ Vectorstore({VECTORSTORE_FILE})가 이미 존재하므로 전처리를 스킵합니다.")
        return VECTORSTORE_FILE

    if not os.path.exists(PREPROCESSED_FILE):
        logging.error(f"❌ 전처리된 데이터 파일({PREPROCESSED_FILE})이 존재하지 않습니다.")
        return None

    try:
        # Info: 파일 읽기
        with open(PREPROCESSED_FILE, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        logging.info(f"📖 {len(sentences)}개의 문장을 읽었습니다.")

        # Info: 각 문장 처리
        processed_data = []
        for i, sentence in enumerate(sentences, 1):
            logging.debug(f"처리 중: [{i}/{len(sentences)}] {sentence}")
            result = process_sentence(sentence)
            processed_data.append(result)
            
        # Info: 결과 저장
        save_to_json(processed_data, VECTORSTORE_FILE)
        logging.info(f"✅ 총 {len(processed_data)}개의 문장이 처리되어 저장되었습니다.")

        return VECTORSTORE_FILE

    except Exception as e:
        logging.error(f"❌ 처리 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    process_data()
