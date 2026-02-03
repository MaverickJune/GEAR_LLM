# DVFS Implementation Summary

## 작업 개요

`nx_client_legacy.py`의 DVFS workflow를 기반으로 `nx_client.py`를 확장하여, LLM inference 중 DQN 기반 CPU frequency control을 수행하도록 구현했습니다.

## 주요 변경 사항

### 1. CPU-Only DVFS
- **GPU frequency 제어 제거**: CPU frequency만 제어
- **Action space**: 13개의 discrete CPU frequency levels (345.6 MHz - 2188.8 MHz)
- **State space**: Power, Temperature, PMU counters (dynamic size)

### 2. 아키텍처
```
Jetson (nx_client.py)          Remote Server (Perf-Trainer)
┌──────────────────┐           ┌──────────────────┐
│ LLM Inference    │           │ DQN Model        │
│ Monitor (P/T/PMU)│◄─ HTTP ──►│ Training         │
│ CPU Freq Control │           │ Action Selection │
└──────────────────┘           └──────────────────┘
```

### 3. 구현 파일

#### Jetson Client
- **[nx_client.py](nx_client.py)**: Main client with DVFS loop
  - LLM generation in subprocess
  - Monitoring and PMU sampling in main process
  - API communication for action selection
  - Periodic model training requests

- **[utils/api_client.py](utils/api_client.py)** (신규): DVFS server API client
  - Model initialization
  - Action retrieval (train/test mode)
  - Training requests
  - Status checking

- **[utils/utils.py](utils/utils.py)**: 
  - `sample_pmu_only()` 함수 추가 - PMU-only sampling
  - 기존 `set_cpu_freq()` 활용

#### Remote Server (Perf-Trainer)
- **[graph/learners.py](Perf-Trainer/graph/learners.py)**: CPU-only DVFS 버전으로 수정
  - `dqn_nx()`: Dynamic state dimension support
  - `get_ob_jetson_llm()`: State extraction for LLM workload
  - `calculate_reward_jetson_llm()`: Power-performance-thermal tradeoff
  - `cal_jetson_power()`: Test metrics calculation
  - Legacy XU3 코드 제거

- **[util_modules/context_module.py](Perf-Trainer/util_modules/context_module.py)**:
  - `learner_factory()` 업데이트: `dqn_nx`, `dqn_pro_nx` 지원

- **[util_modules/dqn_module.py](Perf-Trainer/util_modules/dqn_module.py)**:
  - `learner_factory()` 업데이트

### 4. 문서화
- **[DVFS_README.md](DVFS_README.md)**: 전체 시스템 사용 가이드
  - 아키텍처 설명
  - Setup 방법
  - API 문서
  - Customization 가이드
  - Troubleshooting

- **[Perf-Trainer/README.md](Perf-Trainer/README.md)**: Server 문서
  - API endpoints
  - Deployment 가이드
  - Production setup

- **[Perf-Trainer/requirements.txt](Perf-Trainer/requirements.txt)**: 서버 dependencies

### 5. 테스트
- **[test_dvfs_setup.py](test_dvfs_setup.py)**: 종합 테스트 스크립트
  - Imports verification
  - CPU frequency control test
  - Monitor test
  - PMU counter test
  - API client test (optional)
  - LLM generation test (optional)

## State & Reward

### State Features (Dynamic Size)
```python
state = [
    # Power (per rail, normalized to [0,1])
    rail1_power / 20000,
    rail2_power / 20000,
    ...
    
    # Temperature (per sensor, normalized to [0,1])
    sensor1_temp / 100,
    sensor2_temp / 100,
    ...
    
    # PMU metrics (normalized)
    ipc / 2.0,           # Instructions per cycle
    stall_ratio,         # Stall cycles ratio
]
```

### Reward Function
```python
reward = (
    thermal_penalty * 3.0 +      # Safety (highest priority)
    power_penalty * 2.0 +         # Power efficiency
    performance_reward * 1.0      # Performance
)

# Components:
# - thermal_penalty: -5.0 if temp > 85°C, gradient penalty 70-85°C
# - power_penalty: -power_watts / 10.0
# - performance_reward: ipc * 2.0 - stall_ratio * 1.0
```

## Workflow

### Training Epoch
```python
for sample in samples_while_llm_running:
    # 1. Collect state
    state = {power, temp, pmu}
    
    # 2. Calculate reward from previous state
    reward = calculate_reward(prev_state)
    
    # 3. Get action from server (with exploration)
    action = dvfs_client.get_action(state, train=True)
    
    # 4. Apply frequency
    set_cpu_freq(AVAIL_CPU_FREQS[action])
    
    # 5. Server stores transition in replay buffer
    
    # 6. Periodic training
    if step % TRAIN_STEP == 0:
        dvfs_client.request_update()
        wait_for_training()
```

### Test Epoch
```python
for sample in samples_while_llm_running:
    # Same as training, but:
    # - No exploration (eps=0)
    # - No training
    # - Metrics collected for evaluation
```

## 주요 차이점 (vs legacy)

| Feature | nx_client_legacy.py | nx_client.py (new) |
|---------|---------------------|-------------------|
| Workload | Generic benchmarks | LLM inference (llama.cpp) |
| GPU Control | Yes | No (CPU-only) |
| State Space | Fixed (13 dims) | Dynamic (power + temp + pmu) |
| Action Space | CPU + GPU freqs | CPU freq only (13 levels) |
| Config | INI file | Python constants |
| API | Custom (nn_api) | DVFSClient class |
| Monitor | Legacy Monitor | New Monitor (GearLLM) |

## 사용 방법

### 1. Server 시작 (GPU 머신)
```bash
cd Perf-Trainer
pip install -r requirements.txt
# Edit app.py to set server IP
python app.py
```

### 2. Client 설정 (Jetson)
```bash
# Edit nx_client.py
SERVER_URL = "http://YOUR_SERVER_IP:5000"
AVAIL_CPU_FREQS = [...]  # Your Jetson's frequencies
```

### 3. 테스트
```bash
# Basic tests
python test_dvfs_setup.py

# With server test
python test_dvfs_setup.py http://YOUR_SERVER_IP:5000
```

### 4. 실행
```bash
python nx_client.py
```

## 확장 가능성

### Custom State Features
`get_ob_jetson_llm()` in `learners.py`를 수정하여:
- Utilization metrics 추가
- Memory bandwidth 추가
- Application-specific metrics 추가

### Custom Reward
`calculate_reward_jetson_llm()`를 수정하여:
- Latency constraints 추가
- QoS metrics 반영
- Multi-objective optimization

### Advanced Models
- `dqn_pro_nx`: RNN-based temporal context
- Custom learners: Add to `learner_factory()`

## 검증 사항

- ✅ API client 구현 및 테스트
- ✅ CPU frequency control integration
- ✅ Monitor integration (power, temp)
- ✅ PMU sampling integration
- ✅ State extraction from GearLLM monitor format
- ✅ Reward function for LLM workload
- ✅ Dynamic state dimension support
- ✅ Training/testing loop
- ✅ Documentation (client & server)
- ✅ Test scripts

## 향후 작업

1. **실제 실행 테스트**: Jetson에서 전체 workflow 검증
2. **Hyperparameter tuning**: 
   - Exploration parameters (eps decay)
   - Reward weights
   - Training frequency
3. **성능 측정**: 
   - Power savings vs baseline
   - Performance impact
   - Training convergence
4. **Advanced features**:
   - Multi-agent DVFS (multi-workload)
   - Transfer learning across models
   - Online adaptation

## 파일 목록

### 신규 파일
- `GearLLM/utils/api_client.py` - DVFS API client
- `GearLLM/DVFS_README.md` - 종합 문서
- `GearLLM/test_dvfs_setup.py` - 테스트 스크립트
- `GearLLM/Perf-Trainer/requirements.txt` - Server dependencies
- `GearLLM/IMPLEMENTATION_SUMMARY.md` - 본 문서

### 수정 파일
- `GearLLM/nx_client.py` - DVFS workflow 추가
- `GearLLM/utils/utils.py` - `sample_pmu_only()` 추가
- `GearLLM/Perf-Trainer/graph/learners.py` - CPU-only 버전으로 전환
- `GearLLM/Perf-Trainer/util_modules/context_module.py` - Factory 업데이트
- `GearLLM/Perf-Trainer/util_modules/dqn_module.py` - Factory 업데이트
- `GearLLM/Perf-Trainer/README.md` - 문서 업데이트

## 참고사항

- 모든 코드는 Python 3.x 호환
- Server는 PyTorch GPU 지원 (자동 감지)
- Client는 Jetson에서 CPU-only 실행
- API는 HTTP/JSON 기반 (간단하지만 보안 고려 필요)
- 실제 배포시 HTTPS, authentication 추가 권장
