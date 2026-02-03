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
AVAIL_CPU_FREQS = [
    268800, 729600, 806400, 883200, 960000, 1036800, 1113600, 1190400, 1267200, 
    1344000, 1420800, 1497600, 1574400, 1651200, 1728000, 1804800, 1881600, 
    1958400, 1984000
]

# DVFS Server Configuration
SERVER_URL = "http://192.168.137.1:5000"  # Update with your server IP
API_PREFIX = "/con"  # Use context API (required for dqn_pro_nx)

# Training configuration for DQN Pro with Context
CONTEXT_WARMUP_EPOCHS = 5   # Epochs for collecting context learning data
CONTEXT_SAMPLES_MIN = 500   # Minimum samples before training context
RL_TRAIN_EPOCHS = 15        # Epochs for RL training with context
RL_TRAIN_STEP = 100         # Request RL update every N steps
TEST_EPOCHS = 3             # Final test epochs


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


def run_llm_epoch(prompt, n_predict, n_threads, monitor, cpu, events, dvfs_client, 
                  phase="RECORD_0", sample_callback=None):
    """
    Run one epoch of LLM inference with monitoring and DVFS
    
    Args:
        prompt: LLM prompt
        n_predict: Number of tokens to generate
        n_threads: Number of threads
        monitor: Monitor instance
        cpu: CPU list
        events: PMU event list
        dvfs_client: DVFS client
        phase: Current phase (RECORD_0, RECORD, TEST)
        sample_callback: Optional callback(sample_count) -> bool to check if should continue
        
    Returns:
        dict: Epoch results
    """
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
    print(f"Starting sampling loop (phase: {phase})...")
    while gen_process.is_alive():
        sample_start = time.time()
        
        # Check if callback says to stop early
        if sample_callback and not sample_callback(sample_count):
            print(f"Callback requested early stop at sample {sample_count}")
            break
        
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
            
            # Get action from DVFS server based on phase
            if phase == "RECORD_0":
                # Phase 1: Random exploration for context learning
                response = dvfs_client.get_action(state_data, train=True)
            elif phase == "RECORD":
                # Phase 3: RL training with context
                response = dvfs_client.get_action(state_data, train=True)
            elif phase == "TEST":
                # Phase 5: Testing (no exploration)
                response = dvfs_client.get_action(state_data, train=False)
            else:
                response = None
            
            if response is not None:
                freq_idx = response['action']
                
                # Ensure freq_idx is within bounds
                if 0 <= freq_idx < len(AVAIL_CPU_FREQS):
                    target_freq = AVAIL_CPU_FREQS[freq_idx]
                    
                    # Apply frequency change
                    success, stdout, stderr = set_cpu_freq(target_freq)
                    
                    if success:
                        print(f"Sample {sample_count}: Set CPU freq to {target_freq} kHz (action={freq_idx})")
                    else:
                        print(f"Warning: Failed to set CPU frequency: {stderr}")
                else:
                    print(f"Warning: Invalid frequency index {freq_idx}")
                
                prev_state = state_data
                prev_action = freq_idx
                sample_count += 1
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
    
    # Get generation results
    gen_result = None
    if not generation_result_queue.empty():
        gen_result = generation_result_queue.get()
    
    return {
        'sample_count': sample_count,
        'duration': epoch_end_time - epoch_start_time,
        'gen_result': gen_result
    }


def main():
    """
    Main function for LLM inference with DQN Pro (Context + RL) DVFS control
    
    5-Phase Workflow:
    1. RECORD_0: Warm-up data collection for context learning (random actions)
    2. CONTEXT: Train context model (VRNNCell_V0)
    3. RECORD: RL data collection with context
    4. RL: Train DQN controller
    5. TEST: Evaluate learned policy
    """
    
    # Prompt and generation parameters
    prompt = "What is the meaning of life?"
    n_predict = 50
    n_threads = 4
    
    print("=" * 80)
    print("LLM INFERENCE WITH DQN PRO (CONTEXT + RL) DVFS CONTROL")
    print("=" * 80)
    print(f"Server: {SERVER_URL}")
    print(f"Prompt: {prompt}")
    print(f"Tokens to generate: {n_predict}")
    print(f"Context warm-up epochs: {CONTEXT_WARMUP_EPOCHS}")
    print(f"RL training epochs: {RL_TRAIN_EPOCHS}")
    print(f"Test epochs: {TEST_EPOCHS}")
    print("=" * 80)
    
    # Initialize DVFS client
    dvfs_client = DVFSClient(server_url=SERVER_URL, api_prefix=API_PREFIX)
    
    # Initialize model on remote server (dqn_pro_nx)
    if not dvfs_client.init_model(model_type="dqn_pro_nx"):
        print("ERROR: Failed to initialize DVFS model on server")
        return
    
    # Create monitor instance
    monitor = Monitor()
    
    # ========================================================================
    # PHASE 1: RECORD_0 - Context Learning Data Collection
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: CONTEXT LEARNING DATA COLLECTION (RECORD_0)")
    print("=" * 80)
    print("Collecting temporal data with random exploration...")
    
    total_context_samples = 0
    for epoch in range(1, CONTEXT_WARMUP_EPOCHS + 1):
        print(f"\n--- Warm-up Epoch {epoch}/{CONTEXT_WARMUP_EPOCHS} ---")
        
        result = run_llm_epoch(
            prompt=prompt,
            n_predict=n_predict,
            n_threads=n_threads,
            monitor=monitor,
            cpu=cpu,
            events=events,
            dvfs_client=dvfs_client,
            phase="RECORD_0"
        )
        
        total_context_samples += result['sample_count']
        print(f"Epoch {epoch} completed: {result['sample_count']} samples, "
              f"duration: {result['duration']:.2f}s")
        
        if result['gen_result'] and result['gen_result'].get('is_success', False):
            print(f"Tokens/sec: {result['gen_result']['tokens_per_second']:.2f}")
    
    print(f"\nPhase 1 complete: {total_context_samples} samples collected")
    
    if total_context_samples < CONTEXT_SAMPLES_MIN:
        print(f"WARNING: Only {total_context_samples} samples collected, "
              f"minimum {CONTEXT_SAMPLES_MIN} recommended")
    
    # ========================================================================
    # PHASE 2: CONTEXT - Train Context Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: TRAINING CONTEXT MODEL (VRNN)")
    print("=" * 80)
    print("Training context model on collected temporal sequences...")
    
    # Request context training (this needs to be added to api_client.py)
    # For now, we'll manually send the command
    import requests
    import json
    
    try:
        url = f"{SERVER_URL}{API_PREFIX}/train_context"
        payload = json.dumps({"m_id": dvfs_client.model_id})
        response = requests.post(url, json=payload, timeout=300)  # 5 min timeout
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status', False):
                print("Context model training complete!")
            else:
                print(f"Context training failed: {result}")
        else:
            print(f"Context training request failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error requesting context training: {e}")
        print("Continuing anyway (server might have trained internally)...")
    
    # ========================================================================
    # PHASE 3: RECORD - RL Data Collection with Context
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: RL DATA COLLECTION WITH CONTEXT (RECORD)")
    print("=" * 80)
    print("Collecting RL training data with learned context...")
    
    total_rl_samples = 0
    for epoch in range(1, RL_TRAIN_EPOCHS + 1):
        print(f"\n--- RL Data Epoch {epoch}/{RL_TRAIN_EPOCHS} ---")
        
        result = run_llm_epoch(
            prompt=prompt,
            n_predict=n_predict,
            n_threads=n_threads,
            monitor=monitor,
            cpu=cpu,
            events=events,
            dvfs_client=dvfs_client,
            phase="RECORD"
        )
        
        total_rl_samples += result['sample_count']
        print(f"Epoch {epoch} completed: {result['sample_count']} samples, "
              f"duration: {result['duration']:.2f}s")
        
        if result['gen_result'] and result['gen_result'].get('is_success', False):
            print(f"Tokens/sec: {result['gen_result']['tokens_per_second']:.2f}")
        
        # ====================================================================
        # PHASE 4: RL - Train DQN Controller (periodic)
        # ====================================================================
        if epoch % 5 == 0 or epoch == RL_TRAIN_EPOCHS:
            print(f"\n--- Training DQN Controller (after epoch {epoch}) ---")
            
            try:
                url = f"{SERVER_URL}{API_PREFIX}/train_rl"
                payload = json.dumps({"m_id": dvfs_client.model_id})
                response = requests.post(url, json=payload, timeout=180)
                
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get('status', False):
                        print("DQN training complete!")
                    else:
                        print(f"DQN training failed: {result_data}")
                else:
                    print(f"DQN training request failed: HTTP {response.status_code}")
            except Exception as e:
                print(f"Error requesting DQN training: {e}")
    
    print(f"\nPhase 3+4 complete: {total_rl_samples} RL samples collected and trained")
    
    # ========================================================================
    # PHASE 5: TEST - Evaluate Learned Policy
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: TESTING LEARNED POLICY")
    print("=" * 80)
    print("Evaluating learned policy (no exploration)...")
    
    for epoch in range(1, TEST_EPOCHS + 1):
        print(f"\n--- Test Epoch {epoch}/{TEST_EPOCHS} ---")
        
        result = run_llm_epoch(
            prompt=prompt,
            n_predict=n_predict,
            n_threads=n_threads,
            monitor=monitor,
            cpu=cpu,
            events=events,
            dvfs_client=dvfs_client,
            phase="TEST"
        )
        
        print(f"Test epoch {epoch} completed: {result['sample_count']} samples, "
              f"duration: {result['duration']:.2f}s")
        
        if result['gen_result'] and result['gen_result'].get('is_success', False):
            gen_res = result['gen_result']
            print(f"Generation results:")
            print(f"  Tokens: {gen_res['n_tokens_generated']}")
            print(f"  Total time: {gen_res['total_time_ms']:.2f}ms")
            print(f"  Tokens/sec: {gen_res['tokens_per_second']:.2f}")
    
    # Get final test results from server
    print("\n--- Getting test metrics from server ---")
    try:
        url = f"{SERVER_URL}{API_PREFIX}/get_test_power"
        payload = json.dumps({"m_id": dvfs_client.model_id})
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result_data = response.json()
            if result_data.get('status', False):
                metrics = result_data.get('result', {})
                print(f"Test metrics:")
                print(f"  Average Power: {metrics.get('p', 'N/A')} W")
                print(f"  Duration: {metrics.get('t', 'N/A')} s")
                print(f"  Total Energy: {metrics.get('p_total', 'N/A')} J")
            else:
                print("Failed to get test metrics")
        else:
            print(f"Test metrics request failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error getting test metrics: {e}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING AND TESTING COMPLETE")
    print("=" * 80)
    print(f"Phase 1 (Context data): {total_context_samples} samples")
    print(f"Phase 3 (RL data): {total_rl_samples} samples")
    print(f"Total test epochs: {TEST_EPOCHS}")
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
