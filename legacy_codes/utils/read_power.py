import os, time, csv, math
from pathlib import Path
from typing import Dict, Tuple

try:
    from .utils import read_value
except ImportError:
    from utils import read_value

# =========================
# Config
# =========================
HWMON = Path(os.environ.get("HWMON_PATH", "/sys/class/hwmon/hwmon4"))

HWMON_RAILS = {
    "VDD_IN": ("in1_input", "curr1_input"),
    "VDD_CPU_GPU_CV": ("in2_input", "curr2_input"),
    "VDD_SOC": ("in3_input", "curr3_input"),
}

POLL_MS = int(os.environ.get("HWMON_POLL_MS", "10"))     # default 10ms
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50"))   # print every N samples (0이면 출력 안함)
DURATION_S = float(os.environ.get("DURATION_S", "10"))   # 총 로깅 시간(초)

OUTDIR = Path(os.environ.get("OUTDIR", "."))
CSV_PATH = OUTDIR / os.environ.get("CSV_NAME", "power_log.csv")

# (옵션) DVFS 고정하고 싶으면 사용. 원치 않으면 USE_DVFS=0으로 두면 됨.
USE_DVFS = int(os.environ.get("USE_DVFS", "0"))
CHANGE_FREQ_SCRIPT = os.environ.get("CHANGE_FREQ", "./change_freq_jetson.sh")
CPU_KHZ = os.environ.get("CPU_KHZ", "")   # 예: "1344000"
MIF_HZ  = os.environ.get("MIF_HZ", "")    # Jetson이면 EMC rate. 예: "204000000"

def now_mono() -> float:
    return time.monotonic()

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

def maybe_set_dvfs():
    if not USE_DVFS:
        return
    if not CPU_KHZ or not MIF_HZ:
        raise RuntimeError("USE_DVFS=1이면 CPU_KHZ, MIF_HZ 환경변수를 같이 줘야 함.")
    # sudo -n 사용: 비밀번호 프롬프트 뜨면 실패
    import subprocess
    subprocess.run(["sudo", "-n", "bash", CHANGE_FREQ_SCRIPT, str(CPU_KHZ), str(MIF_HZ)], check=False)
    
def read_temp(num_cpus:int = 7, verbose:bool = False) -> float:
    temp_root = "/sys/devices/virtual/thermal/thermal_zone$$/temp"
    sum = 0.0
    for cpu in range(num_cpus):
        path = temp_root.replace("$$", str(cpu))
        val = read_value(open(path, 'r'))
        if verbose:
            print(f"CPU{cpu} Temp: {val}")
        sum += float(val)
    avg_temp = sum / num_cpus / 1000.0  # Convert to Celsius
    return avg_temp

def main():
    avg_temp = read_temp(verbose=True)
    print(f"Average CPU Temperature: {avg_temp:.2f} °C")

if __name__ == "__main__":
    main()