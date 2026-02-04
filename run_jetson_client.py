import os
import sys
from pathlib import Path
import time
import threading
import argparse
import itertools
import random

import subprocess
from multiprocessing import Process, Pipe, Queue

from GearLLM.jetson_comm_api.comm_api import DQNCommClient

from GearLLM.jetson_utils.state_utils import OrinNaiveStateMonitor
from GearLLM.jetson_utils.dataset_utils import get_hf_dataloader
from GearLLM.jetson_utils.cpu_configs import AVAIL_CPU_FREQ, AVAIL_DQN_CPU_FREQ
from GearLLM.jetson_utils.rewards import cal_cpu_reward
from GearLLM.jetson_utils.set_cpu_freq_utils import set_cpu_frequencies

# Configs for llama.cpp
LLAMA_PATH = "/home/nxc/wjbang/llama.cpp"
MODEL_PATH = "/home/nxc/wjbang/models/Llama-3.2-3B-Instruct-f16.gguf"
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
    
# Main trainer function
def gear_jetson_trainer():
    # Meta Configurations for llama.cpp
    generation_result_queue = Queue()
    INSTRUCTION_PROMPT = "\nSummarize the given article within few sentences."
    N_PREDICT = 200
    N_THREADS = 8
    
    # Client-side training configurations
    TRAINING_TIMEOUT_SEC = 3600 * 2  # Total training time in seconds
    N_ACTIONS = len(AVAIL_DQN_CPU_FREQ)
    TARGET_CPU_UTIL = 0.8
    
    # Dataloader for CNN/DailyMail
    dataloader = get_hf_dataloader(
            dataset_name="abisee/cnn_dailymail",
            name="3.0.0",
            num_samples=300,
            split="test",
            batch_size=1
        )
    
    state_monitor = OrinNaiveStateMonitor()
    comm_client = DQNCommClient(
        server_host="147.46.130.111",  # Replace with actual server IP
        server_port=61103,
        timeout=10.0
    )
    if not comm_client.connect():
        print("Failed to connect to server")
        return
    
    if not comm_client.register_target_util(TARGET_CPU_UTIL):
        print("Failed to register target utilization with server")
        return
    
    # Main training loop
    start_time = time.time()
    monitor_active = True
    
    # Define initial state and action
    prev_state = None
    prev_action = -1
    
    while True:
        if time.time() - start_time > TRAINING_TIMEOUT_SEC:
            print("Training timeout reached. Exiting.")
            # TODO: Allow graceful shutdown to server (save results + model weights)
            break
        
        # Endless iteration over dataloader
        for batch in itertools.cycle(dataloader):
            article = "[article]" + batch[0]['article']
            prompt = article + INSTRUCTION_PROMPT
            
            gen_process = Process(
                target=run_generation,
                args=(generation_result_queue, MODEL_PATH, prompt, N_PREDICT, 
                    True, N_THREADS, False, LIB_PATH)
            )
            gen_process.start()
            
            # Only monitor while generation is running
            while monitor_active:
                if not gen_process.is_alive():
                    monitor_active = False
                    break
                
                curr_state = state_monitor.get_state()
                cpu_utils, cpu_freqs, cpu_temps, total_power = state_monitor.decompose_state(curr_state)
                reward = cal_cpu_reward(cpu_utils, cpu_temps, cluster_num=8, target_util=TARGET_CPU_UTIL) # 8 cores
                
                # Generate an entry for the server-side replay buffer
                new_entry = (prev_state, prev_action, reward, curr_state)
                
                if prev_state is None and prev_action == -1:
                    prev_state = curr_state
                    prev_action = random.randint(0, N_ACTIONS - 1)
                else:
                    action = comm_client.send_state_get_action(curr_state, additional_info={"new_entry": new_entry})
                    prev_state = curr_state
                    prev_action = action
                
                # Apply action (set CPU frequencies)
                target_freq = AVAIL_DQN_CPU_FREQ[prev_action]
                set_cpu_frequencies(target_freq)
                time.sleep(0.1)
                
# Function to test parallel generation
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
    
    