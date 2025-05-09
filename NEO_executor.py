import os
import sys
from ctypes import cdll, c_char_p, c_int

# Note: 해당 모듈은 NEO엔진을 실행하는 모듈임.

class NEOExecutor:
    def __init__(self):
        # 현재 실행 파일이 있는 폴더를 DLL 검색 경로에 추가
        self.dll_path = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(self.dll_path)  # Python 3.8 이상
        sys.path.append(self.dll_path)

        # Windows용 DLL 로드
        self.dll_file = os.path.join(self.dll_path, "NEO", "NeoDLL.dll")
        print(f"🔍 DLL 로드 시도: {self.dll_file}")
        print(f"DLL 파일 존재 여부: {os.path.exists(self.dll_file)}")

        try:
            self.neodll = cdll.LoadLibrary(self.dll_file)
            
            # 함수 매핑
            self.neoInit = self.neodll.NEO_Init
            self.neoExit = self.neodll.NEO_Exit
            self.neoEventEngine = self.neodll.NEO_EventEngine

            # 함수 인자 및 반환 타입 설정
            self.neoEventEngine.argtypes = [c_char_p, c_char_p]
            self.neoEventEngine.restype = c_int

            # 초기화 함수 실행
            print("NEO_Init() 실행")
            init_result = self.neoInit()
            print(f"초기화 결과: {init_result}")

        except Exception as e:
            print(f"DLL 로드 또는 함수 호출 중 오류 발생: {str(e)}")
            if os.name == 'nt':  # Windows인 경우
                try:
                    import subprocess
                    result = subprocess.run(['dumpbin', '/exports', self.dll_file], 
                                         capture_output=True, text=True)
                    print("\n📋 DLL 내보내기 함수 목록:")
                    print(result.stdout)
                except Exception as dep_error:
                    print(f"DLL 분석 실패: {str(dep_error)}")

    def execute_query(self, query: str, result_buffer: str) -> int:
        """
        NEO 엔진에 쿼리를 실행합니다.
        
        Args:
            query: 실행할 쿼리 문자열
            result_buffer: 결과를 저장할 버퍼
            
        Returns:
            실행 결과 코드
        """
        return self.neoEventEngine(
            query.encode('utf-8'),
            result_buffer.encode('utf-8')
        )

    def cleanup(self):
        """NEO 엔진을 종료합니다."""
        if hasattr(self, 'neoExit'):
            self.neoExit()

if __name__ == "__main__":
    try:
        executor = NEOExecutor()
        
        # Info: 테스트 쿼리 실행
        result_buffer = " " * 1024  # Info: 결과를 저장할 버퍼
        result = executor.execute_query("(load-kb \"facts.kb\")", result_buffer)
        print(f"쿼리 실행 결과: {result}")
        
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
    finally:
        if 'executor' in locals():
            executor.cleanup()
