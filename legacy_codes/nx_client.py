import os
from pathlib import Path
import time
import threading
import sys

import subprocess
from multiprocessing import Process, Pipe, Queue

from utils.utils import *
from utils.monitor import Monitor
from utils.api_client import DVFSClient

# Configs for llama.cpp
LLAMA_PATH = "/home/nxc/wjbang/llama.cpp"
MODEL_PATH = "/home/nxc/wjbang/models/Llama-3.2-1B-Instruct-f16.gguf"
LIB_PATH = "/home/nxc/wjbang/llama.cpp/build/lib/libgear_decode.so"
sys.path.insert(0, str(Path(LLAMA_PATH)))
from gear_decode.gear_generate import GearGenerator

# General configs for jetson experiments
cpu=[0, 1, 2, 3, 4, 5, 6, 7]
events = [0,4,5] # ["cycles","stalled-cycles-front", "stalled-cycles-back"]

# Available CPU frequencies for Jetson (in kHz)
# Adjust these based on your specific Jetson device
AVAIL_CPU_FREQS = [
    268800, 729600, 806400, 883200, 960000, 1036800, 1113600, 1190400, 1267200, 1344000, 1420800, 1497600, 1574400, 1651200, 1728000, 1804800, 1881600, 1958400, 1984000
]

# DVFS Server Configuration
SERVER_URL = "http://192.168.137.1:5000"  # Update with your server IP
API_PREFIX = "/con"  # Use context API
TRAIN_STEP = 100  # Request model update every N steps
BENCH_EPOCH = 10  # Number of benchmark epochs
TEST_EPOCH = 3   # Test every N epochs


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
    """
    Main function for LLM inference with DVFS control
    
    Workflow:
    1. Initialize DVFS client and connect to remote server
    2. Run multiple training epochs
    3. For each epoch:
       - Run LLM generation in subprocess
       - Monitor power/temp in main process
       - Collect performance metrics (PMU events)
       - Send state to server and get frequency action
       - Apply frequency changes
       - Train DQN model periodically
    4. Test mode: run without training to evaluate learned policy
    """
    
    # Prompt and generation parameters
    prompt = "What is the meaning of life?"
    n_predict = 50
    n_threads = 4
    
    print("=" * 80)
    print("LLM INFERENCE WITH DVFS CONTROL")
    print("=" * 80)
    print(f"Server: {SERVER_URL}")
    print(f"Prompt: {prompt}")
    print(f"Tokens to generate: {n_predict}")
    print(f"Training epochs: {BENCH_EPOCH}")
    print("=" * 80)
    
    # Initialize DVFS client
    dvfs_client = DVFSClient(server_url=SERVER_URL, api_prefix=API_PREFIX)
    
    # Initialize model on remote server
    if not dvfs_client.init_model(model_type="dqn_nx"):
        print("ERROR: Failed to initialize DVFS model on server")
        return
    
    # Create monitor instance
    monitor = Monitor()
    
    # Training/Testing loop
    count = 0  # Global step counter
    
    for epoch in range(1, BENCH_EPOCH + 1):
        # Determine if this is a training or testing epoch
        TRAIN = (epoch % TEST_EPOCH != 0)
        mode_str = "TRAIN" if TRAIN else "TEST"
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{BENCH_EPOCH} - {mode_str} MODE")
        print(f"{'='*80}")
        
        # Create queues for inter-process communication
        generation_result_queue = Queue()
        
        # Create generation process
        gen_process = Process(
            target=run_generation,
            args=(generation_result_queue, MODEL_PATH, prompt, n_predict, 
                  True, n_threads, False, LIB_PATH)
        )
        
        # Start generation process
        epoch_start_time = time.time()
        gen_process.start()
        
        # Initialize for this epoch
        sample_count = 0
        prev_state = None
        prev_action = None
        
        # Sampling loop while generation is running
        print(f"Starting sampling loop...")
        while gen_process.is_alive():
            sample_start = time.time()
            
            # Sample current state (power, temp, PMU counters)
            try:
                # Query power and temperature
                timestamp, power_data, temp_data = monitor.query()
                
                # Get PMU events
                pmus = sample_pmu_only(events, cpu, 100000)  # 100ms sampling
                
                # Construct state dictionary
                state_data = {
                    'timestamp': timestamp,
                    'power': power_data,
                    'temp': temp_data,
                    'pmu': pmus
                }
                
                # Calculate reward based on previous state
                # Reward = -(power_consumption + performance_penalty)
                reward = 0.0
                if prev_state is not None:
                    # Simple reward: negative power consumption
                    # You can customize this based on your optimization goal
                    avg_power = sum(power_data.values()) / len(power_data) if power_data else 0
                    reward = -avg_power / 1000.0  # Normalize to watts
                
                # Get action from DVFS server
                if TRAIN:
                    # Training mode: get action with exploration
                    response = dvfs_client.get_action(state_data, train=True)
                else:
                    # Test mode: get action without exploration
                    response = dvfs_client.get_action(state_data, train=False)
                
                if response is not None:
                    freq_idx = response['action']
                    
                    # Ensure freq_idx is within bounds
                    if 0 <= freq_idx < len(AVAIL_CPU_FREQS):
                        target_freq = AVAIL_CPU_FREQS[freq_idx]
                        
                        # Apply frequency change
                        success, stdout, stderr = set_cpu_freq(target_freq)
                        
                        if success:
                            print(f"Sample {sample_count}: Set CPU freq to {target_freq} kHz (action={freq_idx}), reward={reward:.4f}")
                        else:
                            print(f"Warning: Failed to set CPU frequency: {stderr}")
                    else:
                        print(f"Warning: Invalid frequency index {freq_idx}")
                    
                    prev_state = state_data
                    prev_action = freq_idx
                    sample_count += 1
                    count += 1
                    
                    # Periodic model update during training
                    if TRAIN and count % TRAIN_STEP == 0:
                        print(f"\n--- Requesting model update (step {count}) ---")
                        if dvfs_client.request_update():
                            # Wait for training to complete
                            print("Waiting for model training...")
                            while not dvfs_client.check_model_status():
                                time.sleep(1)
                            print("Model training complete!")
                        else:
                            print("Warning: Failed to request model update")
                
                else:
                    print("Warning: Failed to get action from server")
                
            except Exception as e:
                print(f"Error during sampling: {e}")
                import traceback
                traceback.print_exc()
            
            # Sleep briefly to avoid overwhelming the system
            elapsed = time.time() - sample_start
            if elapsed < 0.1:  # Min 100ms between samples
                time.sleep(0.1 - elapsed)
        
        # Wait for generation to complete
        gen_process.join()
        epoch_end_time = time.time()
        
        print(f"\nEpoch {epoch} completed: {sample_count} samples collected")
        print(f"Epoch duration: {epoch_end_time - epoch_start_time:.2f} seconds")
        
        # Get generation results
        if not generation_result_queue.empty():
            result = generation_result_queue.get()
            
            if result.get('is_success', False):
                print(f"\n--- Generation Results ---")
                print(f"Output: {result['output_text'][:100]}...")  # First 100 chars
                print(f"Tokens Generated: {result['n_tokens_generated']}")
                print(f"Total Time: {result['total_time_ms']:.2f} ms")
                print(f"Tokens/Second: {result['tokens_per_second']:.2f}")
            else:
                print(f"Error: Generation failed")
                if 'error' in result:
                    print(f"  Error message: {result['error']}")
        
        # For test epochs, get test results from server
        if not TRAIN:
            print("\n--- Test Epoch Results ---")
            test_result = dvfs_client.get_test_power()
            if test_result:
                print(f"Average Power: {test_result.get('p', 'N/A')} W")
                print(f"Average Time: {test_result.get('t', 'N/A')} s")
            else:
                print("Warning: Failed to get test results from server")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total epochs: {BENCH_EPOCH}")
    print(f"Total samples: {count}")
    print(f"Power samples collected: {len(monitor.log_power)}")
    print(f"Temperature samples collected: {len(monitor.log_temp)}")
    
    # Show power/temp statistics
    if monitor.log_power:
        powers = [sum(sample['data'].values()) for sample in monitor.log_power if sample['data']]
        if powers:
            print(f"\nPower Statistics:")
            print(f"  Min: {min(powers):.2f} mW")
            print(f"  Max: {max(powers):.2f} mW")
            print(f"  Avg: {sum(powers)/len(powers):.2f} mW")
    
    if monitor.log_temp:
        temps = [max(sample['data'].values()) for sample in monitor.log_temp if sample['data']]
        if temps:
            print(f"\nTemperature Statistics:")
            print(f"  Min: {min(temps):.2f} °C")
            print(f"  Max: {max(temps):.2f} °C")
            print(f"  Avg: {sum(temps)/len(temps):.2f} °C")
    
    # Cleanup
    dvfs_client.remove_model()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()