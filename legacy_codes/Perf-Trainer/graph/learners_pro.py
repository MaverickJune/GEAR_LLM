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
from graph.model import DQN_AB, ReplayMemory, ReplayMemoryTime, QueueBuffer
from graph.agents import DQN_AGENT_AB
from graph.context import VRNNCell_V0

"""
RL Learners with Context (RNN-based) for CPU-only DVFS
1. provide train/inference service with temporal context
2. logging
3. response to server

For CPU-only DVFS on Jetson with LLM workload:
- State space: power, temperature, PMU counters, CPU utilization
- Context: Learned by VRNNCell_V0 (RNN-based variational autoencoder)
- Action space: CPU frequency index (discrete)
- Reward: CPU-focused with temperature awareness

Transitions dtype:
    <state(np.float32),action(list of int64/long),next_state(np.float32),reward(float)>
"""

# Available CPU Freqs for Jetson (kHz -> normalized to GHz for state space)
JETSON_CPU_FREQS_KHZ = [
    268800, 729600, 806400, 883200, 960000, 1036800, 1113600, 1190400, 1267200, 
    1344000, 1420800, 1497600, 1574400, 1651200, 1728000, 1804800, 1881600, 
    1958400, 1984000
]
JETSON_CPU_FREQS_GHZ = np.array(JETSON_CPU_FREQS_KHZ) / 1e6

def create_log(root, name):
    log_dir = os.path.join(root, name)
    if not os.path.isdir(log_dir): 
        os.makedirs(log_dir)
    log_file = str(datetime.datetime.now().strftime('%m%d-%H%M'))
    log_file = os.path.join(log_dir, log_file) + ".csv"
    return log_file


def get_cpu_utilization(log_data):
    """
    Extract CPU utilization from log data
    Returns list of utilization values for each CPU
    """
    cpu_utils = []
    pmu_data = log_data.get('pmu', {})
    
    # Try to extract utilization from PMU data
    # If direct utilization is available, use it
    # Otherwise, estimate from IPC
    for key, value in pmu_data.items():
        if 'util' in key.lower():
            cpu_utils.append(value)
    
    # If no utilization data, estimate from cycles
    if not cpu_utils:
        cycles_list = []
        for key, value in pmu_data.items():
            if 'cycles' in key.lower() and 'stall' not in key.lower():
                cycles_list.append(value)
        
        # Normalize cycles to approximate utilization [0, 1]
        if cycles_list:
            max_cycles = max(cycles_list) if cycles_list else 1.0
            cpu_utils = [c / max_cycles if max_cycles > 0 else 0.0 for c in cycles_list]
    
    return cpu_utils if cpu_utils else [0.5] * 8  # Default to 8 CPUs


def cal_cpu_reward(cpu_utils, cpu_temps, cluster_num, temp_thre=60):
    """
    Calculate reward for CPU frequency control
    
    Args:
        cpu_utils: List of CPU utilization values [0, 1]
        cpu_temps: List of CPU temperature values (°C)
        cluster_num: Number of CPU clusters
        temp_thre: Temperature threshold (°C)
    
    Returns:
        float: Reward value (higher is better)
    """
    lambda_value = 0.15
    # for cpu
    cpu_u_max, cpu_u_min = 0.85, 0.75
    cpu_u_g = 0.8
    u, v, w = -0.2, 0.21, 0.1
    
    reward_value = 0.0
    cpu_t = cpu_temps[0] if cpu_temps else 50.0  # Use first temp sensor
    
    for cpu_u in cpu_utils:
        # Adjust temperature penalty
        if cpu_t < temp_thre:
            w = 0.2 * math.tanh(temp_thre - cpu_t)
        else:
            w = -2
        
        # Calculate utilization-based reward
        if cpu_u < cpu_u_min or cpu_u > cpu_u_max:
            d = lambda_value
        else:
            d = u + v * math.exp(-(cpu_u - cpu_u_g)**2 / (w ** 2))
        
        reward_value += d
    
    return reward_value / cluster_num if cluster_num > 0 else reward_value


def get_ob_jetson_llm_pro(log_data):
    """
    Extract state and calculate reward for Jetson LLM inference with context learning
    
    State space includes:
    - Power readings (normalized)
    - Temperature readings (normalized)
    - PMU counters (normalized)
    - CPU utilization (for reward calculation)
    
    Args:
        log_data: Dictionary with keys:
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
        cpu_temps = list(temp_data.values())
        max_temp = max(temp_data.values())
        avg_temp = sum(temp_data.values()) / len(temp_data)
    else:
        temp_values = [0.0]
        cpu_temps = [50.0]
        max_temp = 50.0
        avg_temp = 50.0
    
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
        
        # Calculate IPC and stall ratio
        ipc = instructions_total / cycles_total if cycles_total > 0 else 0.0
        stall_ratio = stalls_total / cycles_total if cycles_total > 0 else 0.0
        
        # Normalize PMU metrics
        pmu_values = [
            min(ipc / 2.0, 1.0),
            min(stall_ratio, 1.0)
        ]
    else:
        pmu_values = [0.0, 0.0]
    
    # Extract CPU utilization for reward calculation
    cpu_utils = get_cpu_utilization(log_data)
    
    # Concatenate all state features
    state = np.concatenate([
        power_values,
        temp_values,
        pmu_values
    ]).astype(np.float32)
    
    # Calculate reward using CPU-focused reward function
    cluster_num = len(cpu_utils) if cpu_utils else 1
    reward = cal_cpu_reward(cpu_utils, cpu_temps, cluster_num)
    
    return state, reward


def get_ob(log_data):
    """Legacy wrapper - use get_ob_jetson_llm_pro for new code"""
    return get_ob_jetson_llm_pro(log_data)

def inference(agent, model_c, buffer_i, eps):
    """
    Perform inference with context model
    
    Args:
        agent: DQN agent
        model_c: Context model (VRNNCell_V0)
        buffer_i: Inference buffer with recent states
        eps: Exploration epsilon
        
    Returns:
        state: Full state with context
        actions: Selected action(s)
    """
    # Get current input (last state in buffer)
    model_c.eval()
    agent.eps = eps
    
    with torch.no_grad():
        data = buffer_i[:]
        x = torch.from_numpy(data[-1].state).unsqueeze(0)
        
        # Prepare sequence for context model
        data_seq = [torch.from_numpy(item.state).unsqueeze(0) for item in data]
        
        # Get context from RNN
        context = model_c(data_seq)
        
        # Concatenate current state with context
        state = torch.cat([x, context], dim=1)
        
        # Select action
        actions = agent.select_action(state)
    
    return state, actions

def train_context(model, t_buffer, logger, g_step):
    """
    Train context encoder using variational RNN
    
    Args:
        model: VRNNCell_V0 context model
        t_buffer: ReplayMemoryTime buffer with temporal sequences
        logger: Tensorboard logger
        g_step: Global step for logging
        
    Returns:
        g_step: Updated global step
    """
    # Training hyperparameters
    epochs, b_size, learning_rate = 100, 100, 1e-2
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    train_loader = torch.utils.data.DataLoader(
        t_buffer, 
        shuffle=True, 
        batch_size=b_size, 
        drop_last=True
    )
    
    print(f"Training context model for {epochs} epochs...")
    
    for j in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for i, b in enumerate(train_loader):
            # Extract state sequences from batch
            # b is a batch of temporal sequences
            data = [item.state for item in b]
            
            # Forward pass through VRNN
            recon_loss, kld_loss = model(data)
            loss = recon_loss + kld_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            logger.add_scalar("Train/Context_Total", loss.item(), g_step)
            logger.add_scalar("Train/Context_Recon", recon_loss.item(), g_step)
            logger.add_scalar("Train/Context_KLD", kld_loss.item(), g_step)
            
            epoch_loss += loss.item()
            batch_count += 1
            g_step += 1
        
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        if (j + 1) % 10 == 0:
            print(f"  Epoch {j+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
    
    print("Context model training complete!")
    return g_step

def dqn_pro_nx(pipe, learn_state, init_params):
    """
    Advanced DQN learner with RNN-based context for Jetson CPU DVFS
    
    Workflow:
    1. RECORD_0: Collect data for context learning (warm-up phase)
    2. CONTEXT: Train context model (VRNNCell_V0)
    3. RECORD: Collect data with context for RL training
    4. RL: Train DQN controller
    5. TEST: Evaluate learned policy
    
    State: [raw_features] + [context_features]
    Action: CPU frequency index [0, len(JETSON_CPU_FREQS_KHZ)-1]
    """
    
    # Agent Initialization
    NAME = "DQN_PRO_NX_CPU_LLM"
    ROOT = os.path.join(
        "./db/", NAME, str(datetime.datetime.now().strftime('%m%d-%H%M')))
    log_path = os.path.join(ROOT, "Log")
    model_savepath = os.path.join(ROOT, "Model")
    context_savepath = os.path.join(ROOT, "Context")
    
    if not os.path.isdir(log_path): 
        os.makedirs(log_path)
    if not os.path.isdir(context_savepath):
        os.makedirs(context_savepath)
    
    train_logger = SummaryWriter(log_path)

    # Training hyper-parameters
    EPS_START, EPS_END, EPS_DECAY = 0.99, 0.2, 1000
    n_update, n_batch, SYNC_STEP = 20, 100, 30

    # Task Info
    # State will be determined dynamically, but context adds fixed dimension
    N_X = None  # Raw state dim (determined from first sample)
    N_B = 10    # Context/Belief state dim
    N_A = len(JETSON_CPU_FREQS_KHZ)  # Action space size
    N_H = 25    # Hidden layer size
    N_BUFFER = 12000
    N_W = 10    # Window size for temporal context
    
    print(f"Initializing DQN Pro agent for CPU DVFS with context")
    print(f"Action space: {N_A} frequency levels")
    print(f"Context dimension: {N_B}")
    print(f"Window size: {N_W}")
    
    # Models will be initialized after first state
    contexter = None
    AGENT = None
    
    # Replay Buffers
    context_buffer = ReplayMemoryTime(capacity=N_BUFFER, w=N_W)
    inference_buffer = QueueBuffer(N_W)

    # Reset Training States and Data
    learn_state.value = 1
    prev_state, prev_actions = [None] * 2
    prev_state_full = None
    record_count, test_count, agent_record_count, n_round, g_step = [0] * 5
    context_trained = False

    # Response to start loop
    pipe.send("ready")

    while True:
        # Wait for command
        msg = pipe.recv()
        cmd = msg['cmd']
        print(f"Received pipe message '{cmd}' from process {os.getpid()}")

        if cmd == "RECORD_0":
            """
            Phase 1: Collect data for context learning (warm-up)
            No action selection, just random actions for data collection
            """
            
            # Extract state and reward
            state, reward = get_ob_jetson_llm_pro(msg['data'])
            
            # Initialize context model on first sample
            if contexter is None:
                N_X = len(state)
                print(f"Initializing context model with raw state dim: {N_X}")
                contexter = VRNNCell_V0(x_dim=N_X, z_dim=10, h_dim=N_B)
            
            # Random action selection for exploration
            action = np.random.randint(N_A)
            pipe.send({"action": int(action)})

            # Record in context buffer
            if record_count != 0:
                context_buffer.push(prev_state, prev_actions, state, reward)
            
            prev_state, prev_actions = state, action
            
            # Write log
            if record_count == 0:
                record_log_file = create_log(ROOT, "RECORD_0")
                record_log = open(record_log_file, 'w', newline='')
                record_writer = csv.DictWriter(record_log, msg['data'].keys())
                record_writer.writeheader()
            record_writer.writerow(msg['data'])
            record_log.flush()
            record_count += 1

        elif cmd == "CONTEXT":
            """
            Phase 2: Train context model using collected data
            """
            
            if contexter is None:
                print("Error: Context model not initialized. Need RECORD_0 first.")
                pipe.send({"status": False})
                continue
            
            learn_state.value = 0  # Disable inference
            print(f"\nTraining context model with {len(context_buffer)} samples...")
            
            # Train context model
            g_step = train_context(contexter, context_buffer, train_logger, g_step)
            
            # Save context model
            context_model_path = os.path.join(context_savepath, "context_model.pt")
            torch.save({
                'model_state_dict': contexter.state_dict(),
                'x_dim': N_X,
                'z_dim': 10,
                'h_dim': N_B
            }, context_model_path)
            print(f"Context model saved to {context_model_path}")
            
            # Reset states for RL phase
            prev_state, prev_actions, record_count = None, None, 0
            if 'record_log' in locals() and record_log:
                record_log.close()
            
            # Clear context buffer (optional, or keep for continuous learning)
            # context_buffer = ReplayMemoryTime(capacity=N_BUFFER, w=N_W)
            inference_buffer = QueueBuffer(N_W)
            
            context_trained = True
            learn_state.value = 1
            pipe.send({"status": True})

        elif cmd == "RECORD":
            """
            Phase 3: Collect data with context for RL training
            """
            
            if not context_trained:
                print("Warning: Context not trained yet. Use RECORD_0 and CONTEXT first.")
                # Fall back to random actions
                state, reward = get_ob_jetson_llm_pro(msg['data'])
                action = np.random.randint(N_A)
                pipe.send({"action": int(action)})
                continue
            
            # Extract state and reward
            state, reward = get_ob_jetson_llm_pro(msg['data'])
            
            # Initialize AGENT on first RL record
            if AGENT is None:
                N_S = N_X + N_B  # Full state = raw + context
                print(f"Initializing DQN agent with full state dim: {N_S}")
                AGENT = DQN_AGENT_AB(N_S, N_H, [N_A], N_BUFFER, None)
            
            # Calculate epsilon
            AGENT.eps = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * g_step / EPS_DECAY)

            # Action selection with context
            if len(inference_buffer) == inference_buffer.capacity:
                # Inference with context
                full_state, actions = inference(AGENT, contexter, inference_buffer, AGENT.eps)
                action = actions[0] if isinstance(actions, list) else actions
                
                # Store transition for RL training
                if agent_record_count != 0 and prev_state_full is not None:
                    AGENT.mem.push(prev_state_full, prev_actions, full_state, reward)
                
                prev_state_full = full_state
                agent_record_count += 1
                g_step += 1
            else:
                # Random action while filling inference buffer
                action = np.random.randint(N_A)
            
            pipe.send({"action": int(action)})
            print(f"Step {g_step}: action={action}, eps={AGENT.eps:.3f}, reward={reward:.4f}")

            # Record in inference buffer
            if record_count != 0:
                inference_buffer.push(prev_state, prev_actions, state, reward)

            prev_state, prev_actions = state, action
            
            # Write log
            if record_count == 0:
                record_log_file = create_log(ROOT, "RECORD")
                record_log = open(record_log_file, 'w', newline='')
                record_writer = csv.DictWriter(record_log, msg['data'].keys())
                record_writer.writeheader()
            record_writer.writerow(msg['data'])
            record_log.flush()
            record_count += 1

        elif cmd == "RL":
            """
            Phase 4: Train RL controller
            """
            
            if AGENT is None:
                print("Warning: Cannot train - agent not initialized yet")
                learn_state.value = 1
                pipe.send({"status": False})
                continue
            
            learn_state.value = 0  # Disable inference
            print(f"\nStarting RL training round {n_round}...")
            
            # Train DQN
            losses = AGENT.train(n_round, n_update, n_batch)
            g_step = train_utils.log_scalar_list(train_logger, "Train/RL_Loss", g_step, losses)

            # Reset states
            prev_state, prev_actions, record_count = None, None, 0
            prev_state_full, agent_record_count = None, 0
            if 'record_log' in locals() and record_log:
                record_log.close()

            # Reset inference buffer
            inference_buffer = QueueBuffer(N_W)

            # Save models
            AGENT.save_model(n_round, model_savepath)
            n_round += 1

            if n_round % SYNC_STEP == 0:
                AGENT.sync_model()
            
            print(f"RL training round {n_round-1} complete")
            learn_state.value = 1
            pipe.send({"status": True})

        elif cmd == "TEST":
            """
            Phase 5: Test learned policy
            """
            
            if not context_trained or AGENT is None:
                print("Warning: Models not ready for testing")
                action = np.random.randint(N_A)
                pipe.send({"action": int(action)})
                continue
            
            # Extract state
            state, reward = get_ob_jetson_llm_pro(msg['data'])
            
            # Inference with no exploration
            AGENT.eps = 0

            # Action selection with context
            if len(inference_buffer) == inference_buffer.capacity:
                full_state, actions = inference(AGENT, contexter, inference_buffer, 0.0)
                action = actions[0] if isinstance(actions, list) else actions
            else:
                # Fill buffer first
                action = np.random.randint(N_A)
            
            pipe.send({"action": int(action)})
            
            # Record in inference buffer for next step
            if test_count != 0:
                inference_buffer.push(prev_state, prev_actions, state, reward)
            
            prev_state, prev_actions = state, action
            
            # Write test log
            if test_count == 0:
                test_log_file = create_log(ROOT, "TEST")
                test_log = open(test_log_file, 'w', newline='')
                test_writer = csv.DictWriter(test_log, msg['data'].keys())
                test_writer.writeheader()
            test_writer.writerow(msg['data'])
            test_count += 1
            test_log.flush()

        elif cmd == "END_TEST":
            """
            End test and return results
            """
            
            # Reset test count
            test_count = 0
            if 'test_log' in locals() and test_log:
                test_log.close()
            
            # Calculate metrics
            try:
                power_metrics = cal_jetson_power(test_log_file)
                train_logger.add_scalars("Test", power_metrics, n_round)
                pipe.send(power_metrics)
            except Exception as e:
                print(f"Error calculating test metrics: {e}")
                pipe.send({"t": 0.0, "p": 0.0, "p_total": 0.0})
            
            # Reset inference buffer for next test
            inference_buffer = QueueBuffer(N_W)

        else:
            print(f"Invalid Command: {cmd}")
            pipe.send({"status": False, "error": "Invalid command"})

def cal_jetson_power(log_file):
    """
    Calculate power and time statistics from test log for Jetson
    
    Args:
        log_file: Path to CSV log file
        
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


if __name__ == "__main__":
    """Test code for DQN Pro with context"""
    print("DQN Pro Learners for Jetson CPU DVFS with Context")
    print(f"Available frequencies: {len(JETSON_CPU_FREQS_KHZ)} levels")
    print(f"Range: {JETSON_CPU_FREQS_KHZ[0]/1e6:.2f} - {JETSON_CPU_FREQS_KHZ[-1]/1e6:.2f} GHz")
    
    # Test context model
    N_X = 12  # Example raw state dim
    N_B = 10  # Context dim
    N_W = 10  # Window size
    
    contexter = VRNNCell_V0(x_dim=N_X, z_dim=10, h_dim=N_B)
    print(f"\nContext model: x_dim={N_X}, z_dim=10, h_dim={N_B}")
    
    # Test inference buffer
    inference_buffer = QueueBuffer(N_W)
    for i in range(N_W):
        inference_buffer.push(
            np.ones(N_X, dtype=np.float32) * 0.5,
            np.random.randint(len(JETSON_CPU_FREQS_KHZ)),
            None,
            0.0
        )
    
    print(f"Inference buffer filled: {len(inference_buffer) == inference_buffer.capacity}")
    
    # Test DQN agent
    N_S = N_X + N_B  # Full state
    N_A = len(JETSON_CPU_FREQS_KHZ)
    AGENT = DQN_AGENT_AB(N_S, 25, [N_A], 1000, None)
    print(f"\nDQN agent: state_dim={N_S}, action_dim={N_A}")
    
    # Test inference with context
    AGENT.eps = 0.5
    h, actions = inference(AGENT, contexter, inference_buffer, 0.5)
    print(f"Inference test: context shape={h.shape}, actions={actions}")
    
    print("\nAll tests passed!")

