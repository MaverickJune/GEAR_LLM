# GearLLM with DVFS Control

This system enables Dynamic Voltage and Frequency Scaling (DVFS) for LLM inference on Jetson devices using reinforcement learning (DQN).

## Architecture

The system consists of two main components:

1. **Jetson Client (`nx_client.py`)**: Runs LLM inference and monitors system metrics
2. **Remote Server (`Perf-Trainer/`)**: Hosts DQN models for DVFS control

```
┌─────────────────────────────────────┐
│         Jetson Device               │
│  ┌──────────────────────────────┐   │
│  │     nx_client.py             │   │
│  │  - LLM Inference (llama.cpp) │   │
│  │  - Monitor (Power/Temp/PMU)  │   │
│  │  - CPU Frequency Control     │   │
│  └──────────────────────────────┘   │
│              │ HTTP API              │
└──────────────┼───────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Remote GPU Server              │
│  ┌──────────────────────────────┐   │
│  │    Perf-Trainer (Flask)      │   │
│  │  - DQN Model Training        │   │
│  │  - State Processing          │   │
│  │  - Action Selection          │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Features

### CPU-Only DVFS
- Controls only CPU frequency (GPU frequency is fixed)
- Action space: 13 discrete CPU frequency levels (345.6 MHz - 2188.8 MHz)
- State space: Power, Temperature, PMU counters (cycles, stalls, instructions)

### Reward Function
Optimized for power-performance tradeoff:
- **Power penalty**: Minimize power consumption
- **Performance reward**: Maximize IPC, minimize stalls
- **Thermal penalty**: Avoid overheating (>70°C)

### Training Features
- Experience replay buffer
- ε-greedy exploration with decay
- Periodic model updates
- Test mode for policy evaluation

## Setup

### 1. Jetson Device Setup

```bash
# Install dependencies
cd /home/nxc/wjbang/GearLLM

# Ensure perf_lib is built
cd perf_lib
bash build_perf.sh
cd ..

# Configure CPU frequency access (run once)
sudo chmod 666 /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed
sudo chmod 666 /sys/devices/system/cpu/cpufreq/policy*/scaling_governor

# Set userspace governor
sudo sh perf_lib/change_freq_jetson.sh 1420800
```

### 2. Remote Server Setup

```bash
# Copy Perf-Trainer to GPU server
scp -r GearLLM/Perf-Trainer user@server:/path/to/

# On the server
cd /path/to/Perf-Trainer

# Install dependencies
pip install flask flask-caching torch tensorboard numpy pandas

# Configure server IP in app.py
# Edit line: app.run(host='192.168.137.1', port=5000, ...)
# Change IP to your server's IP address

# Run the server
python app.py
```

### 3. Configure Client

Edit [nx_client.py](nx_client.py):

```python
# Line ~30: Update server URL
SERVER_URL = "http://YOUR_SERVER_IP:5000"

# Line ~25-28: Adjust CPU frequencies for your Jetson
AVAIL_CPU_FREQS = [
    345600, 499200, 652800, ...  # Your Jetson's available frequencies
]

# Line ~31-34: Adjust training parameters
TRAIN_STEP = 100      # Model update frequency
BENCH_EPOCH = 10      # Number of epochs
TEST_EPOCH = 3        # Test every N epochs
```

## Usage

### Start the Server (on GPU machine)

```bash
cd Perf-Trainer
python app.py
```

Server will start on `http://SERVER_IP:5000`

### Run Training on Jetson

```bash
cd /home/nxc/wjbang/GearLLM
python nx_client.py
```

This will:
1. Initialize DQN model on remote server
2. Run LLM inference for multiple epochs
3. Collect power/temp/performance metrics
4. Send states to server and receive frequency actions
5. Apply frequency changes via DVFS
6. Train DQN model periodically
7. Test learned policy every N epochs

### Output

```
================================================================================
LLM INFERENCE WITH DVFS CONTROL
================================================================================
Server: http://192.168.137.1:5000
Prompt: What is the meaning of life?
Tokens to generate: 50
Training epochs: 10
================================================================================

Model initialized successfully with ID: abc-123-def-456

================================================================================
EPOCH 1/10 - TRAIN MODE
================================================================================
Starting sampling loop...
Sample 0: Set CPU freq to 1420800 kHz (action=7), reward=-2.5432
Sample 1: Set CPU freq to 1267200 kHz (action=6), reward=-2.1234
...

--- Requesting model update (step 100) ---
Waiting for model training...
Model training complete!
...

TRAINING COMPLETE
================================================================================
Total epochs: 10
Total samples: 856
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

## File Structure

```
GearLLM/
├── nx_client.py                 # Main client for Jetson
├── utils/
│   ├── api_client.py           # API client for server communication
│   ├── monitor.py              # Power/temperature monitoring
│   ├── utils.py                # CPU frequency control, PMU sampling
│   └── read_power.py           # Power/temp reading functions
├── perf_lib/
│   ├── PyPerf.py               # PMU counter interface
│   └── change_freq_jetson.sh  # Frequency control script
└── Perf-Trainer/               # Remote server (copy to GPU machine)
    ├── app.py                  # Flask server entry point
    ├── graph/
    │   ├── learners.py         # DQN learner (CPU-only DVFS)
    │   ├── agents.py           # DQN agent implementation
    │   └── model.py            # Neural network models
    └── util_modules/
        ├── context_module.py   # Context API (with RNN)
        └── dqn_module.py       # DQN API (vanilla)
```

## API Endpoints

### Context API (`/con/`)
Recommended for complex state processing with RNN context

- `POST /con/init_model` - Initialize model
- `POST /con/get_action` - Get action (training mode)
- `POST /con/get_action_test` - Get action (test mode)
- `GET /con/request_update` - Request model training
- `GET /con/check_model_status` - Check if training complete
- `POST /con/get_test_power` - Get test results
- `GET /con/rm_model` - Remove model

### DQN API (`/nn/`)
Vanilla DQN without context

- Same endpoints as Context API

## Customization

### Adjust State Space

Edit [learners.py](Perf-Trainer/graph/learners.py) function `get_ob_jetson_llm()`:

```python
def get_ob_jetson_llm(log_data):
    # Add/remove state features
    state = np.concatenate([
        power_values,      # Your power features
        temp_values,       # Your temp features
        custom_features,   # Add custom features
        pmu_values
    ]).astype(np.float32)
    
    return state, reward
```

### Adjust Reward Function

Edit [learners.py](Perf-Trainer/graph/learners.py) function `calculate_reward_jetson_llm()`:

```python
def calculate_reward_jetson_llm(total_power_mw, max_temp, avg_temp, ipc, stall_ratio):
    # Customize reward weights
    reward = (
        thermal_penalty * 3.0 +    # Safety weight
        power_penalty * 2.0 +       # Power weight
        performance_reward * 1.0    # Performance weight
    )
    return reward
```

### Adjust Hyperparameters

Edit [learners.py](Perf-Trainer/graph/learners.py) in `dqn_nx()`:

```python
# Exploration
EPS_START = 0.99
EPS_END = 0.2
EPS_DECAY = 1000

# Training
n_update = 20    # Training iterations per update
n_batch = 100    # Batch size
SYNC_STEP = 30   # Target network sync frequency
N_BUFFER = 12000 # Replay buffer size
```

## Troubleshooting

### Cannot set CPU frequency
```bash
# Check permissions
ls -l /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed

# Grant permissions (run once)
sudo chmod 666 /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed
```

### Connection refused to server
- Check server is running: `curl http://SERVER_IP:5000`
- Check firewall: `sudo ufw allow 5000`
- Update `SERVER_URL` in nx_client.py

### Model not initialized
- Check server logs for errors
- Ensure PyTorch is installed on server
- Check model type matches (`dqn_nx` or `dqn_pro_nx`)

### PMU counters not working
```bash
# Rebuild perf library
cd perf_lib
bash build_perf.sh
```

## References

- [llama.cpp](../llama.cpp/) - LLM inference engine
- [Original Perf-Trainer](nx_client_legacy.py) - Legacy multi-device DVFS
- [GearGenerator](../llama.cpp/gear_decode/) - Custom inference wrapper

## License

See main repository LICENSE file.
