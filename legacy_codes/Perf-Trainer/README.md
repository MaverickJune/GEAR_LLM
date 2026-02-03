# Perf-Trainer: DQN-based DVFS Server

Flask-based server for hosting DQN models for Dynamic Voltage and Frequency Scaling (DVFS) control.

## Overview

This server provides RESTful APIs for:
- Initializing and managing DQN models
- Processing system states and returning frequency actions
- Training models with experience replay
- Evaluating learned policies

**Target Platform**: Remote GPU server (not Jetson)  
**Client**: Jetson device running `nx_client.py`

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually
pip install flask flask-caching torch tensorboard numpy pandas
```

### Configuration

Edit `app.py`:

```python
# Line ~20: Set your server IP
app.run(host='192.168.137.1', port=5000, debug=True, threaded=True)
```

### Run Server

```bash
# Development mode
python app.py

# Production mode (recommended)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Server will start on `http://YOUR_IP:5000`

## API Documentation

### Context API (`/con/`)

Recommended API with RNN-based context processing.

#### Initialize Model
```http
POST /con/init_model
Content-Type: application/json

{
  "m_type": "dqn_nx",
  "params": {}
}

Response:
{
  "status": true,
  "m_id": "uuid-string"
}
```

#### Get Action (Training)
```http
POST /con/get_action
Content-Type: application/json

{
  "m_id": "uuid-string",
  "data": {
    "timestamp": 123.456,
    "power": {"rail1": 5000.0, "rail2": 3000.0},
    "temp": {"sensor1": 50.0, "sensor2": 55.0},
    "pmu": {"cpu0_cycles": 1000000, "cpu0_instructions": 800000}
  }
}

Response:
{
  "status": true,
  "action": 7,
  "max_val": 7
}
```

For complete documentation, see [../DVFS_README.md](../DVFS_README.md)

## Model Types

### `dqn_nx`
- **Description**: DQN for Jetson CPU frequency control
- **State**: Power, temperature, PMU counters (dynamic size)
- **Action**: CPU frequency index [0, 12]
- **Reward**: Power-performance-thermal tradeoff

## File Structure

```
Perf-Trainer/
├── app.py                    # Flask server entry
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── util_modules/            # API endpoints
│   ├── cache.py            # Flask caching
│   ├── context_module.py   # /con/* endpoints
│   ├── dqn_module.py       # /nn/* endpoints
│   └── test_module.py      # Test endpoints
│
└── graph/                   # DQN implementation
    ├── learners.py         # Training loops & reward (CPU-only)
    ├── learners_pro.py     # Advanced learners
    ├── agents.py           # DQN agent logic
    ├── model.py            # Neural network models
    ├── context.py          # RNN context models
    └── train_utils.py      # Training utilities
```

## References

- [Full Documentation](../DVFS_README.md)
- [Client Code](../nx_client.py)
