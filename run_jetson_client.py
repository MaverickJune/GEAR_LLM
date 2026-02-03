import os
import sys
from pathlib import Path
import time
import threading
import argparse
import itertools

import subprocess
from multiprocessing import Process, Pipe, Queue
from GearLLM.jetson_utils.state_utils import OrinNaiveStateMonitor
from GearLLM.jetson_utils.dataset_utils import get_hf_dataloader

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

# Maybe can be used later
def gear_jetson_argparser():
    parser = argparse.ArgumentParser(description="GearLLM Jetson Client")
        
def test_parallel_generation():
    instruction_prompt = "\nSummarize the given article within few sentences."
    generation_result_queue = Queue()
    n_predict = 200
    n_threads = 8
    
    dataloader = get_hf_dataloader(
            dataset_name="abisee/cnn_dailymail",
            name="3.0.0",
            num_samples=300,
            split="test",
            batch_size=1
        )
    for batch in itertools.cycle(dataloader):
        article = "[article]" + batch[0]['article']
        prompt = article + instruction_prompt
        # Create generation process
        gen_process = Process(
            target=run_generation,
            args=(generation_result_queue, MODEL_PATH, prompt, n_predict, 
                True, n_threads, False, LIB_PATH)
        )
        
        # Start generation process and monitoring
        start_time = time.time()
        gen_process.start()
        gen_process.join()
        print("Generation process finished.")
        end_time = time.time()
        
        # Get generation results
        if not generation_result_queue.empty():
            result = generation_result_queue.get()
            
            print("\n" + "=" * 80)
            print("GENERATION RESULTS")
            print("=" * 80)
            
            if result.get('is_success', False):
                print(f"\nOutput Text:\n{result['output_text']}")
                print("\n" + "-" * 80)
                print(f"Statistics:")
                print(f"  Tokens Generated: {result['n_tokens_generated']}")
                print(f"  Total Time: {result['total_time_ms']:.2f} ms")
                print(f"  Tokens/Second: {result['tokens_per_second']:.2f}")
                print(f"  Average Time/Token: {result['average_time_per_token']:.2f} ms")
                
                # Show per-token timing
                print(f"\nPer-Token Timing (first 10 tokens):")
                times = result['time_per_token'][:10]
                for i, time_ms in enumerate(times):
                    tps = 1000.0 / time_ms if time_ms > 0 else 0
                    print(f"  Token {i+1:3d}: {time_ms:7.2f} ms ({tps:6.2f} tok/sec)")
                if len(result['time_per_token']) > 10:
                    print(f"  ... and {len(result['time_per_token']) - 10} more tokens")
            else:
                print(f"Error: Generation failed")
                if 'error' in result:
                    print(f"  Error message: {result['error']}")
                print(f"  Error code: {result.get('error_code', 'unknown')}")
        else:
            print("Error: No result received from generation process")
        
        # For testing, break after one iteration
        sys.exit(0)
    
if __name__ == "__main__":
    test_parallel_generation()
    
    