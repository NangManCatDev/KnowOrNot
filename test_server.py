import os
import subprocess
import time
import requests
import sys

# 서버 설정
LLAMA_SERVER_PATH = "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/build/bin/Release/llama-server.exe"
MODEL_PATH = "C:/Users/1/Desktop/wAIfu_llama/llama.cpp/models/downloads/llama2-13b-dpo-test-Q5_K_M.gguf"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
API_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"

# 경로 확인
print(f"서버 경로: {LLAMA_SERVER_PATH}")
print(f"모델 경로: {MODEL_PATH}")

if not os.path.exists(LLAMA_SERVER_PATH):
    print(f"오류: 서버 파일이 존재하지 않습니다: {LLAMA_SERVER_PATH}")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"오류: 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
    sys.exit(1)

# 기존 서버 프로세스 종료 시도
try:
    print("기존 서버 프로세스 종료 시도...")
    os.system(f"taskkill /f /im {os.path.basename(LLAMA_SERVER_PATH)} 2>nul")
    time.sleep(2)
except:
    pass

# 서버 시작 명령어 구성
cmd = [
    LLAMA_SERVER_PATH,
    "-m", MODEL_PATH,
    "--ctx-size", "2048",
    "--host", SERVER_HOST,
    "--port", str(SERVER_PORT),
    "--n-gpu-layers", "100"  # -ngl 대신 --n-gpu-layers 사용
]

cmd_str = " ".join(cmd)
print(f"\n실행 명령어:\n{cmd_str}\n")

# 서버 시작
print("서버 시작 중...")
try:
    # 새 창에서 실행 (Windows)
    # subprocess.Popen(cmd_str, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    # 같은 창에서 실행
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    print("서버 시작 대기 중... (15초)")
    time.sleep(15)
    
    # 서버가 실행 중인지 확인
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print("서버가 시작 후 종료됨")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    
    # API 연결 테스트
    print("\nAPI 연결 테스트 중...")
    for i in range(3):
        try:
            print(f"시도 {i+1}...")
            response = requests.get(f"{API_BASE_URL}/models", timeout=5)
            print(f"상태 코드: {response.status_code}")
            print(f"응답: {response.text}")
            if response.status_code == 200:
                print("서버 연결 성공!")
                break
        except Exception as e:
            print(f"연결 오류: {str(e)}")
        time.sleep(2)
    
    # 간단한 쿼리 테스트
    print("\n간단한 쿼리 테스트...")
    try:
        test_data = {
            "model": "llama2-13b-dpo-test-Q5_K_M",
            "messages": [
                {"role": "user", "content": "안녕하세요?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat/completions", 
            json=test_data,
            timeout=30
        )
        
        print(f"상태 코드: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\n응답:")
            print(result['choices'][0]['message']['content'])
        else:
            print(f"오류 응답: {response.text}")
    
    except Exception as e:
        print(f"쿼리 테스트 오류: {str(e)}")
    
    print("\n서버 실행 중... Ctrl+C로 중단")
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n사용자에 의해 중단됨")
finally:
    # 서버 프로세스 종료
    try:
        process.terminate()
        process.wait(timeout=5)
    except:
        pass
    print("서버 종료됨") 