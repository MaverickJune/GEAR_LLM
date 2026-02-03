import os
import sys
from pathlib import Path
import time
import threading

import subprocess
from multiprocessing import Process, Pipe, Queue
from GearLLM.jetson_utils.state_utils import OrinNaiveStateMonitor

# Configs for llama.cpp
LLAMA_PATH = "/home/nxc/wjbang/llama.cpp"
MODEL_PATH = "/home/nxc/wjbang/models/Llama-3.2-1B-Instruct-f16.gguf"
LIB_PATH = "/home/nxc/wjbang/llama.cpp/build/lib/libgear_decode.so"
sys.path.insert(0, str(Path(LLAMA_PATH)))
from gear_decode.gear_generate import GearGenerator

def run_generation(result_queue, model_path, prompt, n_predict=32, use_instruct=True, 
                   n_threads=8, enable_flash_attn=False, lib_path=LIB_PATH):
    """
    Separate process function to run GearGenerator.generate()
    Puts the result in a Queue for retrieval by parent process
    
    Args:
        result_queue: multiprocessing.Queue to store generation results
        model_path: Path to the GGUF model
        prompt: Input prompt text
        n_predict: Number of tokens to generate
        use_instruct: Use instruct mode
        n_threads: Number of threads
        enable_flash_attn: Enable flash attention
        lib_path: Path to the shared library
    """
    try:
        # Initialize generator in this process
        generator = GearGenerator(lib_path=lib_path)
        
        # Run generation
        result = generator.generate(
            model_path=model_path,
            prompt=prompt,
            n_predict=n_predict,
            use_instruct=use_instruct,
            n_threads=n_threads,
            enable_flash_attn=enable_flash_attn
        )
        
        # Extract all important data from result before it's destroyed
        result_data = {
            'output_text': result.output_text,
            'n_tokens_generated': result.n_tokens_generated,
            'total_time_ms': result.total_time_ms,
            'error_code': result.error_code,
            'is_success': result.is_success,
            'time_per_token': result.time_per_token,
            'token_start_time': result.token_start_time,
            'average_time_per_token': result.average_time_per_token,
            'tokens_per_second': result.tokens_per_second
        }
        
        # Put result in queue
        result_queue.put(result_data)
        
    except Exception as e:
        # Put error in queue
        result_queue.put({
            'error': str(e),
            'is_success': False,
            'error_code': -1
        })