import os
import os.path as path
import csv
import json
import time
import datetime

import math
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

import graph.train_utils as train_utils
from graph.model import DQN_v0, ReplayMemory
from graph.agents import DQN_AGENT

# Available CPU Freqs for Jetson (kHz -> normalized to GHz for state space)
# Adjust based on your Jetson model
JETSON_CPU_FREQS_KHZ = [
    345600, 499200, 652800, 806400, 960000, 1113600, 1267200, 
    1420800, 1574400, 1728000, 1881600, 2035200, 2188800
]
JETSON_CPU_FREQS_GHZ = np.array(JETSON_CPU_FREQS_KHZ) / 1e6  # Convert to GHz for normalization

"""
RL Learners are responsible for 
1. provide train/inference service
2. logging
3. response to server

For CPU-only DVFS on Jetson with LLM workload:
- State space: power, temperature, PMU counters, current frequency
- Action space: CPU frequency index (discrete)
- Reward: optimized for power-performance tradeoff

Transitions dtype:
    <state(np.float32),action(int64/long),next_state(np.float32),reward(float)>
"""
def dqn_nx(pipe, learn_state, init_params):
    """
    DQN learner for Jetson CPU frequency control during LLM inference
    
    State space (dynamic size based on input):
    - Power readings (per rail)
    - Temperature readings (per sensor)
    - PMU counters (cycles, stalls, etc.)
    
    Action space:
    - Discrete: CPU frequency index [0, len(JETSON_CPU_FREQS_KHZ)-1]
    
    Reward:
    - Optimized for power-performance tradeoff
    """
    # Agent Initialization
    NAME = "DQN_NX_CPU_LLM" 
    ROOT = os.path.join(
        "./db/", NAME, str(datetime.datetime.now().strftime('%m%d-%H%M')))
    log_path = os.path.join(ROOT,"Log")
    model_savepath = os.path.join(ROOT,"Model")
    if not os.path.isdir(log_path): os.makedirs(log_path)
    train_logger = SummaryWriter(log_path)

    # training hyper-parameters
    EPS_START = 0.99
    EPS_END = 0.2
    EPS_DECAY = 1000

    n_update, n_batch = 20, 100
    SYNC_STEP = 30
    
    # State and action dimensions
    # State dim will be determined from first sample
    N_S = None  # Will be set dynamically
    N_A = len(JETSON_CPU_FREQS_KHZ)  # Number of frequency levels
    N_BUFFER = 12000
    
    print(f"Initializing DQN agent for CPU DVFS")
    print(f"Action space: {N_A} frequency levels")
    print(f"Available frequencies (GHz): {JETSON_CPU_FREQS_GHZ}")
    
    learn_state.value = 1
    AGENT = None  # Will be initialized after first state
    
    # Reset States
    prev_state, prev_action = [None]*2
    record_count, test_count, n_round, g_step = [0]*4
    
    # Response ready to server
    pipe.send("ready")

    while True:
        # wait for command
        msg = pipe.recv()
        cmd = msg['cmd']
        print("receive pipe message {} from {}".format(cmd, os.getpid()))

        if cmd == "RECORD":
            # Extract state(require np.float32), rewards(float)
            state, reward = get_ob_jetson_llm(msg['data'])
            
            # Initialize agent on first sample (now we know state dimension)
            if AGENT is None:
                N_S = len(state)
                print(f"Initializing agent with state dimension: {N_S}")
                AGENT = DQN_AGENT(N_S, N_A, N_BUFFER, None)
            
            # Inference
            AGENT.eps = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * g_step / EPS_DECAY)
            action = AGENT.select_action(torch.from_numpy(state).unsqueeze(0))
            print(f"Step {g_step}: action={action}, eps={AGENT.eps:.3f}, reward={reward:.4f}")
            pipe.send({"action":int(action)})

            # Add transition (state, action, next_state, reward) into replay buffer
            if record_count != 0:
                AGENT.mem.push(prev_state, prev_action, state, reward)
            prev_state, prev_action = state, action
            g_step += 1

            # Write data of a sample slot
            if record_count == 0:
                record_log_file = create_log(ROOT, "RECORD")
                record_log = open(record_log_file, 'w', newline='')
                record_writer = csv.DictWriter(record_log, msg['data'].keys())
                record_writer.writeheader()
            record_writer.writerow(msg['data'])
            record_log.flush()
            record_count += 1

        elif cmd == "TRAIN":
            if AGENT is None:
                print("Warning: Cannot train - agent not initialized yet")
                learn_state.value = 1
                continue
                
            learn_state.value = 0  # disable inference
            print(f"Starting training round {n_round}...")
            
            # train loop
            losses = AGENT.train(n_round, n_update, n_batch)
            g_step = train_utils.log_scalar_list(train_logger, "Train/Loss", g_step, losses)
            
            # Reset initial states/actions to None
            prev_state, prev_action, record_count = None, None, 0
            if record_log: 
                record_log.close()
            
            # save model
            AGENT.save_model(n_round, model_savepath)
            n_round += 1
            if n_round % SYNC_STEP == 0: 
                AGENT.sync_model()
            
            print(f"Training round {n_round-1} complete")
            learn_state.value = 1
            learn_state.value = 1

        elif cmd == "TEST":
            if AGENT is None:
                print("Warning: Cannot test - agent not initialized yet")
                continue
                
            # Extract state 
            state, reward = get_ob_jetson_llm(msg['data'])
            
            # Inference (no exploration)
            AGENT.eps = 0
            action = AGENT.select_action(torch.from_numpy(state).unsqueeze(0))

            pipe.send({"action": int(action)})
            
            if test_count == 0:
                test_log_file = create_log(ROOT, "TEST")
                test_log = open(test_log_file, 'w', newline='')
                test_writer = csv.DictWriter(test_log, msg['data'].keys())
                test_writer.writeheader()
            test_writer.writerow(msg['data'])
            test_count += 1
            test_log.flush()

        elif cmd == "END_TEST":
            # Reset test count
            test_count = 0
            if test_log: 
                test_log.close()
            
            # Calculate power and time metrics from test log
            power_metrics = cal_jetson_power(test_log_file)
            train_logger.add_scalars("Test", power_metrics, n_round)
            pipe.send(power_metrics)


def create_log(root, name):
    log_dir = os.path.join(root, name)
    if not os.path.isdir(log_dir): 
        os.makedirs(log_dir)
    log_file = str(datetime.datetime.now().strftime('%m%d-%H%M'))
    log_file = os.path.join(log_dir, log_file) + ".csv"
    return log_file


def get_ob_jetson_llm(log_data):
    """
    Extract state and calculate reward for Jetson LLM inference with CPU-only DVFS
    
    State space (normalized to [0, 1] or reasonable ranges):
    - Power readings (per rail, normalized)
    - Temperature readings (per sensor, normalized)
    - PMU counters (normalized)
    
    Reward function:
    - Primary goal: minimize power while maintaining performance
    - Penalize thermal violations
    - Consider PMU metrics for performance indication
    
    Args:
        log_data: Dictionary with keys like:
            - 'timestamp': float
            - 'power': dict with power rail readings (mW)
            - 'temp': dict with temperature sensor readings (°C)
            - 'pmu': dict with PMU counter values
            
    Returns:
        state: np.array (float32) - normalized state vector
        reward: float - reward value
    """
    
    # Extract power data
    power_data = log_data.get('power', {})
    if power_data:
        # Normalize power (assume max ~20W = 20000mW for Jetson)
        power_values = [v / 20000.0 for v in power_data.values()]
        total_power_mw = sum(power_data.values())
    else:
        power_values = [0.0]
        total_power_mw = 0.0
    
    # Extract temperature data
    temp_data = log_data.get('temp', {})
    if temp_data:
        # Normalize temperature (assume range 0-100°C)
        temp_values = [v / 100.0 for v in temp_data.values()]
        max_temp = max(temp_data.values())
        avg_temp = sum(temp_data.values()) / len(temp_data)
    else:
        temp_values = [0.0]
        max_temp = 0.0
        avg_temp = 0.0
    
    # Extract PMU data
    pmu_data = log_data.get('pmu', {})
    pmu_values = []
    
    if pmu_data:
        # Calculate useful metrics from PMU counters
        cycles_total = 0
        instructions_total = 0
        stalls_total = 0
        
        for key, value in pmu_data.items():
            if 'cycles' in key.lower() and 'stall' not in key.lower():
                cycles_total += value
            elif 'instructions' in key.lower():
                instructions_total += value
            elif 'stall' in key.lower():
                stalls_total += value
        
        # Calculate IPC (Instructions Per Cycle) - performance indicator
        ipc = instructions_total / cycles_total if cycles_total > 0 else 0.0
        # Calculate stall ratio - lower is better
        stall_ratio = stalls_total / cycles_total if cycles_total > 0 else 0.0
        
        # Normalize PMU metrics
        pmu_values = [
            min(ipc / 2.0, 1.0),  # IPC typically 0-2, normalize to [0, 1]
            min(stall_ratio, 1.0)  # Stall ratio, cap at 1.0
        ]
    else:
        pmu_values = [0.0, 0.0]
    
    # Concatenate all state features
    state = np.concatenate([
        power_values,
        temp_values,
        pmu_values
    ]).astype(np.float32)
    
    # Calculate reward
    reward = calculate_reward_jetson_llm(
        total_power_mw=total_power_mw,
        max_temp=max_temp,
        avg_temp=avg_temp,
        ipc=pmu_values[0] if pmu_values else 0.0,
        stall_ratio=pmu_values[1] if len(pmu_values) > 1 else 0.0
    )
    
    return state, reward


def calculate_reward_jetson_llm(total_power_mw, max_temp, avg_temp, ipc, stall_ratio):
    """
    Calculate reward for Jetson LLM inference
    
    Reward components:
    1. Power penalty: minimize power consumption
    2. Performance reward: maximize IPC, minimize stalls
    3. Thermal penalty: avoid overheating
    
    Args:
        total_power_mw: Total power consumption in milliwatts
        max_temp: Maximum temperature across sensors (°C)
        avg_temp: Average temperature (°C)
        ipc: Instructions per cycle (normalized)
        stall_ratio: Stall ratio (normalized)
    
    Returns:
        float: Reward value (higher is better)
    """
    
    # Power component: normalize to watts and penalize
    # Target: minimize power while keeping system functional
    power_watts = total_power_mw / 1000.0
    power_penalty = -power_watts / 10.0  # Normalize to reasonable range
    
    # Performance component: reward high IPC and low stalls
    # IPC is already normalized to ~[0, 1]
    performance_reward = ipc * 2.0 - stall_ratio * 1.0
    
    # Thermal component: penalize high temperatures
    # Soft threshold at 70°C, hard threshold at 85°C
    if max_temp > 85.0:
        thermal_penalty = -5.0  # Strong penalty for dangerous temps
    elif max_temp > 70.0:
        thermal_penalty = -(max_temp - 70.0) / 15.0 * 2.0  # Gradual penalty
    else:
        thermal_penalty = 0.0
    
    # Combined reward with weights
    # Prioritize: 1) Safety (thermal), 2) Power efficiency, 3) Performance
    reward = (
        thermal_penalty * 3.0 +      # Weight: 3.0 (safety first)
        power_penalty * 2.0 +         # Weight: 2.0 (power efficiency)
        performance_reward * 1.0      # Weight: 1.0 (performance)
    )
    
    return reward


def cal_jetson_power(log_file):
    """
    Calculate power and time statistics from test log for Jetson
    
    Args:
        log_file: Path to CSV log file with columns including:
            - timestamp
            - power (or individual power rails)
            
    Returns:
        dict: {'t': duration, 'p': average_power_watts, 'p_total': total_energy}
    """
    try:
        df = pd.read_csv(log_file)
        
        if 'timestamp' not in df.columns:
            print(f"Warning: No timestamp column in {log_file}")
            return {"t": 0.0, "p": 0.0, "p_total": 0.0}
        
        timestamps = df['timestamp'].values
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        
        # Try to extract power data
        # Look for 'power' column or reconstruct from individual rails
        if 'power' in df.columns:
            power_mw = df['power'].values
        else:
            # Try to find power-related columns
            power_cols = [col for col in df.columns if 'power' in col.lower() or '_w' in col.lower()]
            if power_cols:
                power_mw = df[power_cols].sum(axis=1).values
            else:
                print(f"Warning: No power columns found in {log_file}")
                return {"t": duration, "p": 0.0, "p_total": 0.0}
        
        # Convert to watts
        power_w = power_mw / 1000.0
        
        # Calculate average power
        avg_power = np.mean(power_w)
        
        # Calculate total energy (integrate power over time)
        if len(timestamps) > 1:
            total_energy = np.trapz(power_w, timestamps)
        else:
            total_energy = avg_power * duration
        
        return {
            "t": float(duration),
            "p": float(avg_power),
            "p_total": float(total_energy)
        }
        
    except Exception as e:
        print(f"Error calculating power from {log_file}: {e}")
        return {"t": 0.0, "p": 0.0, "p_total": 0.0}


# Legacy functions (kept for backward compatibility)
def get_ob(log_data):
    """Legacy function - use get_ob_jetson_llm instead"""
    s = np.random.random(13).astype(np.float32)
    r = 1.0
    return s, r


if __name__ == "__main__":
    print("DQN Learners for Jetson CPU DVFS")
    print(f"Available frequencies: {len(JETSON_CPU_FREQS_KHZ)} levels")
    print(f"Range: {JETSON_CPU_FREQS_KHZ[0]/1e6:.2f} - {JETSON_CPU_FREQS_KHZ[-1]/1e6:.2f} GHz")

