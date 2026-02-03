# nx_client_pro.py - DQN Pro with Context Learning

Advanced DVFS client using `dqn_pro_nx` with VRNNCell_V0 context learning for temporal modeling.

## Overview

`nx_client_pro.py` implements a 5-phase training workflow:

1. **RECORD_0**: Collect warm-up data for context learning (random exploration)
2. **CONTEXT**: Train VRNNCell_V0 to learn temporal patterns
3. **RECORD**: Collect RL data with learned context
4. **RL**: Train DQN controller on augmented state (raw + context)
5. **TEST**: Evaluate learned policy

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                  nx_client_pro.py                      │
│                                                        │
│  Phase 1: RECORD_0 (warm-up)                          │
│    └─► Random actions, collect temporal sequences     │
│                                                        │
│  Phase 2: CONTEXT                                     │
│    └─► Train VRNNCell_V0 on sequences                │
│                                                        │
│  Phase 3: RECORD (RL data)                            │
│    └─► Context-aware actions, collect transitions    │
│                                                        │
│  Phase 4: RL (periodic)                               │
│    └─► Train DQN on [raw_state + context]            │
│                                                        │
│  Phase 5: TEST                                        │
│    └─► Evaluate policy (no exploration)              │
└────────────────────────────────────────────────────────┘
```

## Configuration

Edit the following parameters in `nx_client_pro.py`:

```python
# Server configuration
SERVER_URL = "http://192.168.137.1:5000"
API_PREFIX = "/con"  # Must use context API

# Training phases
CONTEXT_WARMUP_EPOCHS = 5    # Epochs for context data collection
CONTEXT_SAMPLES_MIN = 500    # Minimum samples before training context
RL_TRAIN_EPOCHS = 15         # Epochs for RL training
RL_TRAIN_STEP = 100          # Periodic RL update frequency
TEST_EPOCHS = 3              # Final test epochs

# Available CPU frequencies (adjust for your Jetson)
AVAIL_CPU_FREQS = [
    268800, 729600, ..., 1984000  # kHz
]
```

## Usage

### 1. Start Server (GPU Machine)

```bash
cd Perf-Trainer
python app.py
```

### 2. Run Training (Jetson)

```bash
cd /home/nxc/wjbang/GearLLM
python nx_client_pro.py
```

### Expected Output

```
================================================================================
LLM INFERENCE WITH DQN PRO (CONTEXT + RL) DVFS CONTROL
================================================================================
Server: http://192.168.137.1:5000
Prompt: What is the meaning of life?
Context warm-up epochs: 5
RL training epochs: 15
Test epochs: 3
================================================================================

Model initialized successfully with ID: abc-123-def

================================================================================
PHASE 1: CONTEXT LEARNING DATA COLLECTION (RECORD_0)
================================================================================
Collecting temporal data with random exploration...

--- Warm-up Epoch 1/5 ---
Starting sampling loop (phase: RECORD_0)...
Sample 0: Set CPU freq to 1420800 kHz (action=10)
Sample 1: Set CPU freq to 960000 kHz (action=4)
...
Epoch 1 completed: 42 samples, duration: 8.52s
Tokens/sec: 5.87

[... epochs 2-5 ...]

Phase 1 complete: 215 samples collected

================================================================================
PHASE 2: TRAINING CONTEXT MODEL (VRNN)
================================================================================
Training context model on collected temporal sequences...
Context model training complete!

================================================================================
PHASE 3: RL DATA COLLECTION WITH CONTEXT (RECORD)
================================================================================
Collecting RL training data with learned context...

--- RL Data Epoch 1/15 ---
Starting sampling loop (phase: RECORD)...
Sample 0: Set CPU freq to 1267200 kHz (action=8)
...

--- Training DQN Controller (after epoch 5) ---
DQN training complete!

[... continues through epoch 15 ...]

Phase 3+4 complete: 645 RL samples collected and trained

================================================================================
PHASE 5: TESTING LEARNED POLICY
================================================================================
Evaluating learned policy (no exploration)...

--- Test Epoch 1/3 ---
...
Test epoch 1 completed: 43 samples, duration: 8.75s
Tokens: 50, Total time: 8524.32ms, Tokens/sec: 5.86

--- Getting test metrics from server ---
Test metrics:
  Average Power: 4.523 W
  Duration: 25.6 s
  Total Energy: 115.8 J

================================================================================
TRAINING AND TESTING COMPLETE
================================================================================
Phase 1 (Context data): 215 samples
Phase 3 (RL data): 645 samples
Total test epochs: 3
Power samples collected: 1024
Temperature samples collected: 1024

Power Statistics:
  Min: 3245.23 mW
  Max: 8932.12 mW
  Avg: 5234.67 mW

Temperature Statistics:
  Min: 42.50 °C
  Max: 68.75 °C
  Avg: 55.32 °C
```

## Differences from nx_client.py

| Feature | nx_client.py | nx_client_pro.py |
|---------|--------------|------------------|
| Model | dqn_nx | dqn_pro_nx |
| Phases | 1 (RL only) | 5 (Context + RL) |
| State | Raw only | Raw + Context (10-dim) |
| Training | Continuous | Staged (warm-up → context → RL) |
| Complexity | Lower | Higher |
| Performance | Good | Better for dynamic workloads |

## Training Workflow Details

### Phase 1: RECORD_0 (Warm-up)

- **Purpose**: Collect temporal data for context learning
- **Actions**: Random exploration (no learned policy)
- **Duration**: `CONTEXT_WARMUP_EPOCHS` epochs
- **Data**: Stored in `ReplayMemoryTime` with window size 10
- **Server Mode**: RECORD_0 command

### Phase 2: CONTEXT

- **Purpose**: Learn temporal patterns from warm-up data
- **Model**: VRNNCell_V0 (Variational RNN)
- **Training**: 100 epochs on temporal sequences
- **Output**: 10-dimensional belief state (context vector)
- **Server Mode**: CONTEXT command

### Phase 3: RECORD (RL Data)

- **Purpose**: Collect RL transitions with context
- **Actions**: Context-aware (VRNN + DQN inference)
- **Duration**: `RL_TRAIN_EPOCHS` epochs
- **State**: Concatenate [raw_state, context]
- **Server Mode**: RECORD command

### Phase 4: RL (Periodic)

- **Purpose**: Train DQN on augmented state space
- **Frequency**: Every 5 epochs (or configurable)
- **Training**: Standard DQN with experience replay
- **Server Mode**: RL command

### Phase 5: TEST

- **Purpose**: Evaluate learned policy
- **Actions**: No exploration (epsilon = 0)
- **Duration**: `TEST_EPOCHS` epochs
- **Server Mode**: TEST command

## State and Reward

### Raw State
- Power readings (per rail, normalized)
- Temperature readings (per sensor, normalized)
- PMU counters (IPC, stall ratio, normalized)

### Context Vector
- 10-dimensional belief state from VRNNCell_V0
- Encodes temporal patterns and workload dynamics

### Full State (Input to DQN)
```python
full_state = [raw_state] + [context_vector]
# Dimension: N_X + 10 (e.g., 15 + 10 = 25)
```

### Reward Function
Uses CPU-focused reward with utilization target:
- Target utilization: 0.8 (range 0.75-0.85)
- Temperature threshold: 60°C
- Gaussian-shaped reward around target
- Heavy penalty for overheating

## Troubleshooting

### Context model not training
- **Symptom**: Phase 2 fails or takes too long
- **Solution**: 
  - Increase `CONTEXT_WARMUP_EPOCHS`
  - Check server logs for errors
  - Ensure enough memory on server

### RL not improving
- **Symptom**: Actions stay random, no convergence
- **Solution**:
  - Verify context trained successfully
  - Increase `RL_TRAIN_EPOCHS`
  - Check reward function (should vary)
  - Inspect tensorboard logs

### Out of memory on server
- **Solution**:
  - Reduce buffer size in `learners_pro.py`: `N_BUFFER = 6000`
  - Reduce batch size: `n_batch = 50`
  - Reduce window size: `N_W = 5`

### Connection timeout
- **Symptom**: HTTP timeouts during CONTEXT or RL phases
- **Solution**:
  - Increase timeout in `api_client.py`
  - Training phases are long (5-10 minutes normal)
  - Check server didn't crash

## API Endpoints Used

### Standard Endpoints
- `POST /con/init_model` - Initialize dqn_pro_nx
- `POST /con/get_action` - Get action (all phases)
- `GET /con/rm_model` - Clean up

### DQN Pro Specific
- `POST /con/train_context` - Train context model (Phase 2)
- `POST /con/train_rl` - Train RL controller (Phase 4)
- `POST /con/get_test_power` - Get test metrics (Phase 5)

## Performance Expectations

### Training Time
- Phase 1: ~5-10 minutes (5 epochs)
- Phase 2: ~5-15 minutes (context training)
- Phase 3+4: ~30-45 minutes (15 epochs + 3 RL updates)
- Phase 5: ~5 minutes (3 test epochs)
- **Total**: ~45-75 minutes

### Memory Usage
- Jetson: ~500MB (client + LLM)
- Server: ~2-4GB (PyTorch + buffers)

### Results
- Power savings: 10-20% vs random policy
- Performance: Similar to baseline (within 5%)
- Temperature: Reduced peaks, better thermal management

## When to Use

**Use nx_client_pro.py when:**
- Complex, dynamic LLM workloads
- Temporal patterns important (variable token generation)
- Have GPU server for training
- Want best performance
- Can afford longer training time

**Use nx_client.py when:**
- Simple workloads
- Quick training desired
- Limited server resources
- Good enough performance acceptable

## References

- [learners_pro.py](Perf-Trainer/graph/learners_pro.py) - Server-side implementation
- [context.py](Perf-Trainer/graph/context.py) - VRNNCell_V0 model
- [DQN_PRO_README.md](Perf-Trainer/DQN_PRO_README.md) - Detailed documentation
- [VRNN Paper](https://arxiv.org/abs/1506.02216) - Original VRNN paper

## License

See main repository LICENSE file.
