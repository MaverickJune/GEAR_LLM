from . import freq_measurement_utils, power_measurement_utils, stall_measurement_utils

class OrinNaiveStateMonitor:
    def __init__(self):
        self.stall_monitor = stall_measurement_utils.OrinPMCMonitor()
        self.stall_monitor.get_utilization()
        
    def _get_per_core_utilization(self):
        per_cpu = self.stall_monitor.get_utilization()
        active_cpu_utils = []
        for cpu, data in per_cpu.items():
            active_cpu_utils.append(data['active_pct'])
        return active_cpu_utils
    
    def _get_per_policy_cpu_freqencies(self):
        freqs = freq_measurement_utils.read_cpu_frequencies_by_policy()
        return [freqs.get('policy0'), freqs.get('policy4')]
    
    def _get_per_core_temperatures(self):
        temps = power_measurement_utils.read_temp(num_cpus=8, verbose=False, return_each=True)
        return temps
    
    def _get_total_power(self):
        power = power_measurement_utils.read_total_power_w()
        return power
    
    def get_state(self):
        state = []
        state.extend(self._get_per_core_utilization())  # 8
        state.extend(self._get_per_policy_cpu_freqencies())  # 2
        state.extend(self._get_per_core_temperatures())  # 8
        state.append(self._get_total_power())  # 1
        return state  # total 19 dimensions