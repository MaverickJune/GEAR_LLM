from GearLLM.jetson_utils.state_utils import OrinNaiveStateMonitor
from GearLLM.jetson_utils.freq_measurement_utils import read_cpu_frequencies_by_policy
from GearLLM.jetson_utils.set_cpu_freq_utils import set_cpu_frequencies
from GearLLM.jetson_utils.cpu_configs import AVAIL_CPU_FREQ

monitor = OrinNaiveStateMonitor()
state = monitor.get_state()
print("State vector:", state)

target_freq = AVAIL_CPU_FREQ[6]  # 예: 4번째 주파수 선택
print(f"Setting CPU frequencies to {target_freq / 1000:.0f} MHz")
set_results = set_cpu_frequencies(target_freq)
freq = read_cpu_frequencies_by_policy()
print("Current CPU frequencies by policy:", freq)
