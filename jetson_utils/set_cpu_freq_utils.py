import os

def set_cpu_frequencies(target_freq_khz: int, verbose: bool = False) -> dict[str, bool]:
    """
    모든 CPU policy의 주파수를 target_freq로 설정합니다.
    
    Args:
        target_freq_khz: 목표 주파수 (kHz 단위)
    
    Returns:
        dict: {"policy0": success, "policy4": success} 형태
    """
    policies = {
        "policy0": "/sys/devices/system/cpu/cpufreq/policy0",
        "policy4": "/sys/devices/system/cpu/cpufreq/policy4",
    }
    
    results = {}
    
    for policy_name, policy_path in policies.items():
        try:
            # 1. governor를 userspace로 변경
            governor_path = os.path.join(policy_path, "scaling_governor")
            with open(governor_path, 'w') as f:
                f.write("userspace")
            
            # 2. 목표 주파수 설정
            setspeed_path = os.path.join(policy_path, "scaling_setspeed")
            with open(setspeed_path, 'w') as f:
                f.write(str(target_freq_khz))
            
            results[policy_name] = True
            if verbose:
                print(f"  {policy_name}: Set to {target_freq_khz / 1000:.0f} MHz")
            
        except PermissionError:
            results[policy_name] = False
            print(f"  {policy_name}: Permission denied (run with sudo)")
        except FileNotFoundError as e:
            results[policy_name] = False
            print(f"  {policy_name}: Path not found - {e}")
        except Exception as e:
            results[policy_name] = False
            print(f"  {policy_name}: Error - {e}")
    
    return results


def set_cpu_freq_range(policy_id: int, min_freq_khz: int, max_freq_khz: int) -> bool:
    """
    특정 policy의 주파수 범위를 설정합니다.
    
    Args:
        policy_id: Policy 번호 (0 또는 4)
        min_freq_khz: 최소 주파수 (kHz)
        max_freq_khz: 최대 주파수 (kHz)
    
    Returns:
        bool: 성공 여부
    """
    policy_path = f"/sys/devices/system/cpu/cpufreq/policy{policy_id}"
    
    try:
        # 순서 중요: max를 먼저 올린 후 min 설정
        with open(os.path.join(policy_path, "scaling_max_freq"), 'w') as f:
            f.write(str(max_freq_khz))
        
        with open(os.path.join(policy_path, "scaling_min_freq"), 'w') as f:
            f.write(str(min_freq_khz))
        
        return True
    except (PermissionError, FileNotFoundError) as e:
        print(f"Error setting freq range for policy{policy_id}: {e}")
        return False