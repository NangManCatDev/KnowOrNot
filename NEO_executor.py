import os
import sys
from ctypes import cdll, c_char_p, c_int, create_string_buffer

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

    def execute_query(self, query: str, result_buffer: str = None):
        """
        NEO 엔진에 쿼리를 실행합니다.
        
        Args:
            query: 실행할 쿼리 문자열
            result_buffer: 결과를 저장할 버퍼 (기본값: None)
            
        Returns:
            (실행 결과 코드, 결과 문자열) 튜플
        """
        print(f"🔍 실행할 쿼리: {query}")
        print(f"📝 쿼리 바이트: {query.encode('utf-8')}")
        
        query_bytes = query.encode('utf-8')
        
        # 기본 버퍼 크기 설정
        if result_buffer is None:
            result_buffer = " " * 1024
            
        buffer_bytes = create_string_buffer(len(result_buffer) + 1)  # +1 for null terminator
        
        result = self.neoEventEngine(query_bytes, buffer_bytes)
        return result, buffer_bytes.value.decode('utf-8')

    def cleanup(self):
        """NEO 엔진을 종료합니다."""
        if hasattr(self, 'neoExit'):
            self.neoExit()
            
    def load_kb_file(self, kb_file_path):
        """
        KB 파일을 로드하고 각 줄을 실행합니다.
        
        Args:
            kb_file_path: KB 파일의 경로
            
        Returns:
            성공 여부
        """
        try:
            print(f"KB 파일 로드 시도: {kb_file_path}")
            print(f"KB 파일 존재 여부: {os.path.exists(kb_file_path)}")
            
            # 파일을 직접 열어서 각 줄을 실행
            with open(kb_file_path, 'r', encoding='utf-8') as kb_file:
                file_contents = kb_file.read()
                print(f"KB 파일 내용 미리보기: {file_contents[:100]}...")
                
                lines = file_contents.splitlines()
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith(';'):  # 빈 줄이나 주석 건너뛰기
                        continue
                    
                    # 각 줄을 NEO 엔진에 전달
                    result, output = self.execute_query(line)
                    if result != 1:  # 오류 발생 시 
                        print(f"  라인 실행 오류: {line} -> {result}, {output}")
                        return False
                
            print(f"KB 파일 '{os.path.basename(kb_file_path)}' 로드 완료")
            return True
            
        except FileNotFoundError:
            print(f"KB 파일을 찾을 수 없음: {kb_file_path}")
            return False
        except Exception as e:
            print(f"KB 파일 로드 중 오류 발생: {str(e)}")
            return False

if __name__ == "__main__":
    try:
        # 현재 작업 디렉토리 출력
        cwd = os.getcwd()
        print(f"현재 작업 디렉토리: {cwd}")
        
        executor = NEOExecutor()
        
        # KB 파일 경로 정의 (두 곳 모두 시도)
        kb_paths = [
            os.path.join(cwd, "facts.kb"),               # 현재 디렉토리
            os.path.join(cwd, "NEO", "facts.kb")         # NEO 디렉토리
        ]
        
        # 로드 성공 여부
        load_success = False
        
        # 각 경로에서 파일 로드 시도
        for kb_path in kb_paths:
            if executor.load_kb_file(kb_path):
                load_success = True
                print(f"KB 파일 로드 성공: {kb_path}")
                break
        
        if not load_success:
            print("모든 KB 파일 로드 시도 실패")
            
        # 이제 다른 명령어를 실행할 수 있습니다
        print("\n작업 완료. NEO 엔진이 준비되었습니다.")
        
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
    finally:
        if 'executor' in locals():
            executor.cleanup()
