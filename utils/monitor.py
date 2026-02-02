'''
TODO: jetson 에 맞는 방식으로 전력 계속 읽어오기 (done, read_power.py 참고)
TODO: jetson 에 맞는 방식으로 온도 계속 읽어오기 (done, read_power.py 참고)
TODO: 모든 subprocess와 process가 같은 timeline으로 로깅해야함
'''
import time
from .utils import set_cpu_freq, sample
from .read_power import read_hwmon_power_w, read_temp, read_total_power_w

class Monitor:
    def __init__(self):
        self.log_temp = []
        self.log_power = []
    
    def query(self):
        """
        Query current power and temperature, log with timestamp.
        Uses time.time() for cross-process synchronization with C++ subprocess
        that uses std::chrono::system_clock.
        """
        # Use time.time() - same as C++ std::chrono::system_clock
        # This ensures sync with subprocess timeline
        timestamp = time.time()
        
        try:
            power_data = read_total_power_w()
            temp_data = read_temp()
            
            self.log_power.append({
                'timestamp': timestamp,
                'data': power_data
            })
            
            self.log_temp.append({
                'timestamp': timestamp,
                'data': temp_data
            })
            
            return timestamp, power_data, temp_data
        except Exception as e:
            # If reading fails, log error but continue
            print(f"Warning: Failed to query sensors: {e}")
            return timestamp, {}, {}