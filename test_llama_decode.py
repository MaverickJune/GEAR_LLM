import sys
from pathlib import Path

LLAMA_PATH = "/home/nxc/wjbang/llama.cpp"
sys.path.insert(0, str(Path(LLAMA_PATH)))

from gear_decode.gear_generate import GearGenerator


# 설정
MODEL_PATH = "/home/nxc/wjbang/models/Llama-3.2-1B-Instruct-f16.gguf"
LIB_PATH = "/home/nxc/wjbang/llama.cpp/build/lib/libgear_decode.so"

# Generator 초기화
generator = GearGenerator(lib_path=LIB_PATH)

# 텍스트 생성
result = generator.generate(
    model_path=MODEL_PATH,
    prompt="What is the capital of France?",
    n_predict=50,
    use_instruct=True,
    n_threads=8
)

# 결과 출력
if result.is_success:
    print(f"생성된 텍스트:\n{result.output_text}")
    print(f"\n성능: {result.tokens_per_second:.2f} tokens/sec")
else:
    print(f"오류 발생 (코드: {result.error_code})")