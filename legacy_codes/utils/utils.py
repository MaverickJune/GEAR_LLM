import subprocess
import time
import os
import sys

# perf_lib 경로를 동적으로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
perf_lib_path = os.path.join(parent_dir, 'perf_lib')
if perf_lib_path not in sys.path:
    sys.path.insert(0, perf_lib_path)

import PyPerf as Perf

def sample(config, monitor, events, cpus, t):
    raw = Perf.sys_perf(cpus, events, int(t))
    log_data = {}
    log_data = monitor.query()
    return log_data, raw

def sample_pmu_only(events, cpus, t):
    """
    Sample PMU counters only without monitor
    
    Args:
        events: List of event indices
        cpus: List of CPU indices
        t: Sampling time in microseconds
        
    Returns:
        Dictionary with PMU counter values
    """
    raw = Perf.sys_perf(cpus, events, int(t))
    abbrevs = Perf.get_supported_abbrevs()
    
    result_dict = {}
    for i, c in enumerate(cpus):
        for j, e in enumerate(events):
            name = f"cpu{c}_{abbrevs[e]}"
            result_dict[name] = raw[i][j]
    
    return result_dict

def test_sample():
    """
    Test PMU sampling with online CPUs only
    Uses a shorter sampling time (100ms) for quick testing
    """
    try:
        # 온라인 상태인 CPU만 가져오기
        online_cpus = []
        num_cpus = os.cpu_count()
        for i in range(num_cpus):
            try:
                with open(f'/sys/devices/system/cpu/cpu{i}/online', 'r') as f:
                    if f.read().strip() == '1':
                        online_cpus.append(i)
            except FileNotFoundError:
                # CPU0는 online 파일이 없을 수 있음 (항상 온라인)
                if i == 0:
                    online_cpus.append(i)
        
        if not online_cpus:
            online_cpus = [0]  # 최소한 CPU0는 사용
        
        print(f"Testing with online CPUs: {online_cpus}")
        
        events = [0]  # CPU cycles만 측정
        abbrevs = Perf.get_supported_abbrevs()
        
        # 100ms만 측정 (1000000 -> 100000 마이크로초)
        pmus = Perf.sys_perf(online_cpus, events, 100000)
        
        result_dict = {}
        for i, c in enumerate(online_cpus):
            for j, e in enumerate(events):
                name = "cpu{}_{}".format(c, abbrevs[e])
                result_dict[name] = pmus[i][j]
        print(result_dict)
        return result_dict
    except Exception as e:
        print(f"Error in test_sample: {e}")
        import traceback
        traceback.print_exc()
        return None

# CPU Cores Utilization
def parse_core_util(prev_cpu_time, num_cpu):
    last_idles, last_totals = prev_cpu_time   
    with open('/proc/stat') as f: 
        lines = f.readlines()
    utils = []
    for i, l in enumerate(lines[1:num_cpu+1]):
        fields = [float(column) for column in l.strip().split()[1:]]
        idle, total = fields[3], sum(fields)
        idle_delta, total_delta = idle - last_idles[i], total - last_totals[i]
        last_idles[i], last_totals[i] = idle, total
        utilization = 1.0 - idle_delta / total_delta
        utils.append(utilization)
    return utils, (last_idles, last_totals)

def get_core_time(num_cpu):
    with open('/proc/stat') as f: 
        lines = f.readlines()
    idles,totals = [0]*num_cpu, [0]*num_cpu
    for i, l in enumerate(lines[1:num_cpu+1]):
        fields = [float(column) for column in l.strip().split()[1:]]
        idles[i] = fields[3]
        totals[i] = sum(fields)
    return idles, totals

def check_cpus(config):
    cpu_num = int(config['cpu']['num'])
    online_cpu_num = 0
    for i in range(cpu_num):
        if get_value(config['cpu']['on'].replace("$$",str(i)))=="1":
            online_cpu_num += 1
    return online_cpu_num

# File operations for sysfs nodes
def get_value(file):
    with open(file, 'r') as f:
        text = f.read().strip("\n")
    return text

def read_value(f):
    f.seek(0)
    text = f.read().strip("\n")
    return text

def set_value(file,v):
    with open(file,'w') as f:
        f.write(str(v))
    return 0

def s2i(s):
    # convert string to int
    i = int(s.replace(',',''))
    return i

def set_cpu_freq(cpu_freq_khz):
    """
    Jetson CPU frequency를 설정하는 함수
    
    Args:
        cpu_freq_khz (int or str): CPU frequency in kHz (예: 1984000)
    
    Returns:
        tuple: (success: bool, output: str, error: str)
    
    Example:
        success, output, error = set_cpu_freq(1984000)
        if success:
            print("CPU frequency set successfully")
        else:
            print(f"Failed to set CPU frequency: {error}")
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    script_path = os.path.join(parent_dir, 'perf_lib', 'change_freq_jetson.sh')
    
    if not os.path.exists(script_path):
        return False, "", f"Script not found: {script_path}"
    
    try:
        # sudo 권한으로 스크립트 실행
        cmd = ['sudo', 'sh', script_path, str(cpu_freq_khz)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)