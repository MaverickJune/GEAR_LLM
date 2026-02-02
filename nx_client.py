import os
from pathlib import Path
import time
import threading

import subprocess
from multiprocessing import Process, Pipe, Queue

from utils.utils import *
from utils.monitor import Monitor

# Configs for llama.cpp
LLAMA_PATH = "/home/nxc/wjbang/llama.cpp"
MODEL_PATH = "/home/nxc/wjbang/models/Llama-3.2-1B-Instruct-f16.gguf"
LIB_PATH = "/home/nxc/wjbang/llama.cpp/build/lib/libgear_decode.so"
sys.path.insert(0, str(Path(LLAMA_PATH)))
from gear_decode.gear_generate import GearGenerator

# General configs for jetson experiments
cpu=[0, 1, 2, 3, 4, 5, 6, 7]
events = [0,4,5] # ["cycles","stalled-cycles-front", "stalled-cycles-back"]


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

# Main function
def main():
    # Create queues for inter-process communication
    generation_result_queue = Queue()
    
    # Prompt and generation parameters
    prompt = "What is the meaning of life?"
    n_predict = 50
    n_threads = 4
    
    print(f"Starting parallel execution:")
    print(f"  - LLM Generation: {prompt}")
    print(f"  - DVFS Monitoring")
    print("=" * 80)
    
    # Create monitor instance in main process
    monitor = Monitor()
    
    # Create generation process
    gen_process = Process(
        target=run_generation,
        args=(generation_result_queue, MODEL_PATH, prompt, n_predict, 
              True, n_threads, False, LIB_PATH)
    )
    
    # Start generation process and monitoring
    start_time = time.time()
    gen_process.start()
    
    # Run monitoring loop in main process
    print("Monitoring started in main process")
    monitoring_active = True
    while monitoring_active:
        # Check if generation is still running
        if not gen_process.is_alive():
            monitoring_active = False
            break
        
        # Query power and temperature
        monitor.query()
        time.sleep(0.1)
    
    # Wait for generation to complete (if not already done)
    gen_process.join()
    print("Monitoring stopped")
    
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
    
    print("\n" + "=" * 80)
    print("MONITORING RESULTS")
    print("=" * 80)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Access monitor data directly
    log_power = monitor.log_power
    log_temp = monitor.log_temp
    
    print(f"Power samples collected: {len(log_power)}")
    print(f"Temperature samples collected: {len(log_temp)}")
    
    # Show some power/temp statistics if available
    if log_power:
        print("\nPower Statistics:")
        print(f"  Samples: {len(log_power)}")
        print(f"  First sample timestamp: {log_power[0]['timestamp']:.3f}")
        print(f"  Last sample timestamp: {log_power[-1]['timestamp']:.3f}")
        print(f"  Duration: {log_power[-1]['timestamp'] - log_power[0]['timestamp']:.3f} seconds")
        
    if log_temp:
        print("\nTemperature Statistics:")
        print(f"  Samples: {len(log_temp)}")
        print(f"  First sample timestamp: {log_temp[0]['timestamp']:.3f}")
        print(f"  Last sample timestamp: {log_temp[-1]['timestamp']:.3f}")
        print(f"  Duration: {log_temp[-1]['timestamp'] - log_temp[0]['timestamp']:.3f} seconds")
    
    print("=" * 80)
    
    return result if not generation_result_queue.empty() else None


if __name__ == "__main__":
    main()