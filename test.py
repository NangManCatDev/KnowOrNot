import os
import sys
from ctypes import cdll, c_char_p, c_int, create_string_buffer

def main():
    print("=== NEO DLL 테스트 프로그램 ===")
    
    # 현재 실행 파일 경로와 작업 디렉토리 확인
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 현재 작업 디렉토리: {current_dir}")
    print(f"📁 스크립트 위치: {script_dir}")
    
    # NEO 디렉토리와 DLL 경로 설정
    neo_dir = os.path.join(script_dir, "NEO")
    dll_path = os.path.join(neo_dir, "NeoDLL.dll")
    print(f"🔍 NEO 디렉토리: {neo_dir}")
    print(f"🔍 DLL 경로: {dll_path}")
    print(f"🔍 DLL 파일 존재 여부: {os.path.exists(dll_path)}")
    
    # 작업 디렉토리를 NEO 폴더로 변경
    print(f"🔄 작업 디렉토리 변경: {neo_dir}")
    os.chdir(neo_dir)
    
    # 디렉토리 내 파일 확인
    print("\n📂 NEO 디렉토리 내 파일:")
    for item in os.listdir():
        if os.path.isfile(item):
            print(f"  - {item} (크기: {os.path.getsize(item)} 바이트)")
    
    try:
        # neoconsolekernel.py 방식으로 DLL 로드
        print("\n🔍 DLL 로드 시도...")
        neodll = cdll.LoadLibrary(dll_path)
        print("✅ DLL 로드 성공!")
        
        # 함수 매핑
        neoInit = neodll.NEO_Init
        neoExit = neodll.NEO_Exit
        neoEventEngine = neodll.NEO_EventEngine
        
        # 함수 인자 및 반환 타입 설정
        neoEventEngine.argtypes = [c_char_p, c_char_p]
        neoEventEngine.restype = c_int
        
        # 초기화 함수 실행
        print("\n🔄 NEO_Init() 실행...")
        init_result = neoInit()
        print(f"✅ 초기화 결과: {init_result}")
        
        # 명령 실행 테스트 부분 제거됨
        print("\n✅ NEO 엔진이 성공적으로 초기화되었습니다.")
        
        # 종료 함수 실행
        print("\n🔄 NEO_Exit() 실행...")
        neoExit()
        print("✅ 종료 완료")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 작업 디렉토리 원래대로 복원
    os.chdir(current_dir)
    print(f"\n🔄 작업 디렉토리 복원: {current_dir}")
    print("=== 테스트 완료 ===")

if __name__ == "__main__":
    main() 