# DQN Pro with Context Learning for DVFS

Advanced DQN learner with RNN-based temporal context for Jetson CPU frequency control.

## Overview

`dqn_pro_nx` extends the vanilla DQN with a Variational Recurrent Neural Network (VRNN) to learn temporal context from system state sequences. This enables the agent to make frequency decisions based on both current state and historical patterns.

## Architecture

```
┌─────────────────────────────────────────────────┐
│               dqn_pro_nx                        │
│                                                 │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │ Raw State    │────────►│ VRNNCell_V0     │  │
│  │ Sequence     │         │ (Context Model) │  │
│  │ (window=10)  │         └────────┬────────┘  │
│  └──────────────┘                  │           │
│                                    │           │
│                           Context Vector       │
│                                    │           │
│  ┌──────────────┐                  │           │
│  │ Current      │                  │           │
│  │ Raw State    │──────────────────┴──────┐    │
│  └──────────────┘                         │    │
│                                            ▼    │
│                              ┌──────────────────┤
│                              │ DQN Agent        │
│                              │ (Action Branch)  │
│                              └────────┬─────────┘
│                                       │          │
│                                  Action (freq)   │
└───────────────────────────────────────┼─────────┘
                                        │
                                   CPU Freq
```

## Training Workflow

### Phase 1: Context Learning (RECORD_0 → CONTEXT)

1. **Collect temporal data** (RECORD_0):
   - Run workload with random frequency selection
   - Collect state sequences for context learning
   - Store in `ReplayMemoryTime` buffer (window-based)
   
2. **Train context model** (CONTEXT):
   - Train VRNNCell_V0 on collected sequences
   - Learn to encode temporal patterns into belief state
   - Save context model

### Phase 2: RL Training (RECORD → RL)

3. **Collect RL data with context** (RECORD):
   - Use trained context model
   - State = [raw_state] + [context_vector]
   - DQN learns on augmented state space
   
4. **Train DQN** (RL):
   - Standard DQN training with experience replay
   - Actions based on full state (raw + context)

### Phase 3: Testing (TEST)

5. **Evaluate policy** (TEST):
   - No exploration (epsilon = 0)
   - Use both context and DQN models

## Usage

### API Workflow

For clients using the Context API (`/con/`):

```python
# 1. Initialize model
response = dvfs_client.init_model(model_type="dqn_pro_nx")

# 2. Phase 1: Collect context data
for i in range(warm_up_steps):
    response = dvfs_client.get_action(state_data, train=True)
    # Server internally uses RECORD_0 mode
    apply_frequency(response['action'])

# 3. Train context model
dvfs_client.request_context_training()
wait_for_context_training()

# 4. Phase 2: RL training
for epoch in range(training_epochs):
    for step in range(steps_per_epoch):
        response = dvfs_client.get_action(state_data, train=True)
        apply_frequency(response['action'])
    
    # Periodic RL updates
    if epoch % update_freq == 0:
        dvfs_client.request_update()
        wait_for_training()

# 5. Testing
for step in range(test_steps):
    response = dvfs_client.get_action(state_data, train=False)
    apply_frequency(response['action'])
```

### Direct API Endpoints

The Context API supports these commands via pipe:

- `RECORD_0`: Collect data for context learning
- `CONTEXT`: Train context model
- `RECORD`: Collect data with context for RL
- `RL`: Train DQN controller
- `TEST`: Evaluate policy
- `END_TEST`: Get test results

## State and Reward

### Raw State (Input to Context Model)

Same as vanilla DQN:
- Power readings (normalized)
- Temperature readings (normalized)
- PMU counters (IPC, stall ratio)

### Context Vector

10-dimensional belief state learned by VRNN:
- Encodes temporal patterns
- Captures workload dynamics
- Represents system "momentum"

### Full State (Input to DQN)

```python
full_state = concatenate([raw_state, context_vector])
# Dimension: N_X + N_B (e.g., 12 + 10 = 22)
```

### Reward Function

Uses CPU-focused reward:

```python
def cal_cpu_reward(cpu_utils, cpu_temps, cluster_num):
    """
    Reward based on:
    - CPU utilization target (0.75 - 0.85)
    - Temperature threshold (<60°C)
    - Gaussian-shaped reward around target utilization
    """
    lambda_value = 0.15
    cpu_u_max, cpu_u_min = 0.85, 0.75
    cpu_u_g = 0.8  # Target utilization
    u, v, w = -0.2, 0.21, 0.1
    temp_thre = 60
    
    reward = 0.0
    for cpu_u in cpu_utils:
        if cpu_t < temp_thre:
            w = 0.2 * tanh(temp_thre - cpu_t)
        else:
            w = -2  # Heavy penalty for overheating
        
        if cpu_u < cpu_u_min or cpu_u > cpu_u_max:
            d = lambda_value
        else:
            d = u + v * exp(-(cpu_u - cpu_u_g)^2 / w^2)
        
        reward += d
    
    return reward / cluster_num
```

## Hyperparameters

### Context Model (VRNNCell_V0)
- `x_dim`: Raw state dimension (auto-detected)
- `z_dim`: 10 (latent dimension)
- `h_dim`: 10 (hidden/belief state dimension)
- Window size: 10 (temporal sequence length)

### DQN Agent
- State dim: `N_X + N_B` (raw + context)
- Action dim: 13 (CPU frequency levels)
- Hidden dim: 25
- Replay buffer: 12000
- Batch size: 100
- Update frequency: 20 iterations
- Target sync: Every 30 rounds

### Exploration
- Epsilon start: 0.99
- Epsilon end: 0.2
- Epsilon decay: 1000 steps

## Files Structure

```
Perf-Trainer/graph/
├── learners_pro.py         # This module
├── context.py              # VRNNCell_V0 implementation
├── agents.py               # DQN_AGENT_AB
└── model.py                # Neural network models
```

## Comparison: DQN vs DQN Pro

| Feature | `dqn_nx` | `dqn_pro_nx` |
|---------|----------|--------------|
| Context | None | VRNN (10-dim) |
| State | Raw only | Raw + Context |
| Training Phases | 1 (RL only) | 2 (Context + RL) |
| Memory | Single buffer | Context + RL buffers |
| Temporal Modeling | None | Explicit (VRNN) |
| Complexity | Lower | Higher |
| Performance | Good | Better for dynamic workloads |

## When to Use

**Use `dqn_nx` when:**
- Simple workloads
- Static patterns
- Limited computational resources
- Quick training desired

**Use `dqn_pro_nx` when:**
- Complex, dynamic workloads (e.g., LLM inference)
- Temporal dependencies important
- Have GPU for training
- Want best performance

## Example: Full Training Script

```python
# nx_client_pro.py
from utils.api_client import DVFSClient

# Initialize
dvfs_client = DVFSClient(
    server_url="http://192.168.137.1:5000",
    api_prefix="/con"
)

dvfs_client.init_model(model_type="dqn_pro_nx")

# Phase 1: Context learning (warm-up)
print("Phase 1: Collecting context data...")
for epoch in range(5):  # Warm-up epochs
    run_llm_with_random_freq()  # Custom function
    # Server collects RECORD_0 data

# Train context
print("Training context model...")
dvfs_client.request_context_training()
while not dvfs_client.check_context_status():
    time.sleep(1)

# Phase 2: RL training
print("Phase 2: RL training with context...")
for epoch in range(20):
    run_llm_with_dvfs()  # Uses context + DQN
    
    if epoch % 5 == 0:
        dvfs_client.request_update()  # Train RL
        while not dvfs_client.check_model_status():
            time.sleep(1)

# Phase 3: Testing
print("Phase 3: Testing...")
for epoch in range(3):
    results = run_llm_test()
    print(f"Epoch {epoch}: {results}")

test_metrics = dvfs_client.get_test_power()
print(f"Final results: {test_metrics}")
```

## Troubleshooting

### Context model not learning
- Increase warm-up data: Collect more RECORD_0 samples
- Check state normalization: Ensure features are properly scaled
- Adjust VRNN hyperparameters: Try different z_dim or h_dim

### RL not improving
- Verify context is trained: Check context model saved
- Increase exploration: Higher epsilon or longer decay
- Check reward function: Ensure it's not constant

### Out of memory
- Reduce buffer size: `N_BUFFER = 6000`
- Smaller batch size: `n_batch = 50`
- Shorter window: `N_W = 5`

## References

- [VRNN Paper](https://arxiv.org/abs/1506.02216) - Chung et al., 2015
- [DQN Paper](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Vanilla DQN Implementation](learners.py) - dqn_nx
- [Context Model](context.py) - VRNNCell_V0

## License

See main repository LICENSE file.
