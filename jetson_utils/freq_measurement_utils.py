import os
from typing import Optional

def read_cpu_frequencies() -> dict[int, Optional[int]]:
    """
    Jetson Orin NX의 8개 CPU 코어 주파수를 읽어옵니다.
    
    Returns:
        dict: {cpu_id: frequency_khz} 형태의 딕셔너리
              읽기 실패 시 해당 코어는 None
    """
    frequencies = {}
    
    for cpu_id in range(8):
        freq_path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_cur_freq"
        
        try:
            with open(freq_path, 'r') as f:
                frequencies[cpu_id] = int(f.read().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            frequencies[cpu_id] = None
    
    return frequencies


def read_cpu_frequencies_by_policy() -> dict[str, Optional[int]]:
    """
    Policy 단위로 CPU 주파수를 읽어옵니다.
    Jetson Orin NX: policy0 (cpu0-3), policy4 (cpu4-7)
    
    Returns:
        dict: {"policy0": freq, "policy4": freq} 형태
    """
    policies = {
        "policy0": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "policy4": "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
    }
    
    frequencies = {}
    for policy_name, freq_path in policies.items():
        try:
            with open(freq_path, 'r') as f:
                frequencies[policy_name] = int(f.read().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            frequencies[policy_name] = None
    
    return frequencies


def get_cpu_freq_info(cpu_id: int = 0) -> dict[str, Optional[int]]:
    """
    특정 CPU의 상세 주파수 정보를 읽어옵니다.
    
    Args:
        cpu_id: CPU 번호 (0-7)
    
    Returns:
        dict: cur_freq, min_freq, max_freq (kHz 단위)
    """
    base = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq"
    
    info = {}
    for name in ["scaling_cur_freq", "scaling_min_freq", "scaling_max_freq"]:
        try:
            with open(os.path.join(base, name), 'r') as f:
                info[name.replace("scaling_", "")] = int(f.read().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            info[name.replace("scaling_", "")] = None
    
    return info


if __name__ == "__main__":
    # 모든 코어 주파수 출력
    print("=== CPU Frequencies (all cores) ===")
    freqs = read_cpu_frequencies()
    for cpu_id, freq in freqs.items():
        if freq:
            print(f"  cpu{cpu_id}: {freq / 1000:.0f} MHz")
        else:
            print(f"  cpu{cpu_id}: N/A")
    
    # Policy 단위 출력
    print("\n=== CPU Frequencies (by policy) ===")
    policy_freqs = read_cpu_frequencies_by_policy()
    for policy, freq in policy_freqs.items():
        if freq:
            print(f"  {policy}: {freq / 1000:.0f} MHz")
    
    # 상세 정보
    print("\n=== CPU0 Detailed Info ===")
    info = get_cpu_freq_info(0)
    for key, val in info.items():
        if val:
            print(f"  {key}: {val / 1000:.0f} MHz")
