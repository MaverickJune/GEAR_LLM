import os, time, csv, math
from pathlib import Path
from typing import Dict, Tuple

# Variables for power measurement
HWMON = Path(os.environ.get("HWMON_PATH", "/sys/class/hwmon/hwmon4"))
HWMON_RAILS = {
    "VDD_IN": ("in1_input", "curr1_input"),
    "VDD_CPU_GPU_CV": ("in2_input", "curr2_input"),
    "VDD_SOC": ("in3_input", "curr3_input"),
}

'''
sudo chmod a+r /sys/class/hwmon/hwmon4/*_input
사용전 해당 command를 입력하여 권한설정
'''
def read_hwmon_power_w() -> Dict[str, float]:
    """
    Returns power in Watts for each rail.
    Assumes in*_input is mV, curr*_input is mA (INA3221 typical sysfs).
    """
    p = {}
    for rail, (vfile, cfile) in HWMON_RAILS.items():
        v_mv = int((HWMON / vfile).read_text().strip())
        c_ma = int((HWMON / cfile).read_text().strip())
        p[rail] = (v_mv / 1000.0) * (c_ma / 1000.0)
    return p

def read_total_power_w() -> float:
    p = read_hwmon_power_w()
    return sum(p.values())

def read_temp(num_cpus: int = 8, verbose: bool = False, return_each: bool = True) -> float:
    temp_root = "/sys/devices/virtual/thermal/thermal_zone$$/temp"
    sum = 0.0
    cpu_temps = []
    for cpu in range(num_cpus):
        path = temp_root.replace("$$", str(cpu))
        with open(path, 'r') as f:
            val = f.read().strip()
        if verbose:
            print(f"CPU{cpu} Temp: {val}")
        t_val = float(val) / 1000.0  # Convert to Celsius
        cpu_temps.append(t_val)
        sum += t_val
    avg_temp = sum / num_cpus  # Convert to Celsius
    if not return_each:
        return avg_temp
    else:
        return cpu_temps


if __name__ == "__main__":
    avg_temp = read_temp(verbose=False, return_each=False)
    print(f"Average CPU Temperature: {avg_temp:.2f} °C")
    
    cpu_temps = read_temp(verbose=False, return_each=True)
    for i, t in enumerate(cpu_temps):
        print(f"CPU{i} Temperature: {t:.2f} °C")
        
    total_power = read_total_power_w()
    print(f"Total Power Consumption: {total_power:.3f} W")