import os
import struct
import time
from ctypes import *

# ============================================
# PMC Event Codes for Cortex-A78AE
# ============================================

PERF_TYPE_HARDWARE = 0
PERF_TYPE_RAW = 4

# ARM Cortex-A78AE (Jetson Orin NX)
# Reference: ARM Cortex-A78AE TRM, Chapter 11 PMU
EVENTS = {
    'cycles': (PERF_TYPE_HARDWARE, 0x0),      # CPU_CYCLES
    'stalled_frontend': (PERF_TYPE_RAW, 0x23),  # STALL_FRONTEND
    'stalled_backend': (PERF_TYPE_RAW, 0x24),   # STALL_BACKEND
}

# 추가로 유용한 이벤트들
EXTRA_EVENTS = {
    'instructions': (PERF_TYPE_HARDWARE, 0x1),  # INST_RETIRED
    'cache_misses': (PERF_TYPE_RAW, 0x03),      # L1D_CACHE_REFILL
    'branch_miss': (PERF_TYPE_RAW, 0x10),       # BR_MIS_PRED
    'mem_access': (PERF_TYPE_RAW, 0x13),        # MEM_ACCESS
}


# ============================================
# perf_event structures
# ============================================

class perf_event_attr(Structure):
    _fields_ = [
        ('type', c_uint),
        ('size', c_uint),
        ('config', c_ulong),
        ('sample_period', c_ulong),
        ('sample_type', c_ulong),
        ('read_format', c_ulong),
        ('flags', c_ulong),
        ('wakeup_events', c_uint),
        ('bp_type', c_uint),
        ('bp_addr', c_ulong),
        ('bp_len', c_ulong),
    ]

NR_perf_event_open = 241  # ARM64 syscall number


def perf_event_open(event_type, config, cpu=-1, group_fd=-1, flags=0):
    attr = perf_event_attr()
    attr.type = event_type
    attr.size = sizeof(perf_event_attr)
    attr.config = config
    attr.flags = 0
    
    libc = CDLL('libc.so.6', use_errno=True)
    fd = libc.syscall(NR_perf_event_open, byref(attr), -1, cpu, group_fd, flags)
    
    if fd < 0:
        errno = get_errno()
        raise OSError(errno, f"perf_event_open failed: {os.strerror(errno)}")
    return fd


# ============================================
# PMC Monitor Class
# ============================================

class OrinPMCMonitor:
    def __init__(self, cpus=None, events=None):
        """
        Args:
            cpus: 모니터링할 CPU 리스트 (None이면 모든 코어)
            events: 모니터링할 이벤트 dict (None이면 기본 3개)
        """
        if cpus is None:
            # Orin NX: 8 cores (6 Cortex-A78AE + 2 reserved or 8 A78AE depending on SKU)
            cpus = list(range(os.cpu_count()))
        
        if events is None:
            events = EVENTS
        
        self.cpus = cpus
        self.events = events
        self.fds = {}  # {cpu: {event_name: fd}}
        self.prev_values = {}
        
        self._open_counters()
    
    def _open_counters(self):
        for cpu in self.cpus:
            self.fds[cpu] = {}
            for name, (etype, config) in self.events.items():
                try:
                    fd = perf_event_open(etype, config, cpu)
                    self.fds[cpu][name] = fd
                except OSError as e:
                    print(f"[CPU {cpu}] {name} unavailable: {e}")
    
    def read_counters(self):
        """모든 CPU의 현재 PMC 값 읽기"""
        values = {}
        for cpu in self.cpus:
            values[cpu] = {}
            for name, fd in self.fds[cpu].items():
                data = os.read(fd, 8)
                values[cpu][name] = struct.unpack('Q', data)[0]
        return values
    
    def get_utilization(self):
        """
        각 CPU의 active/stall 비율 계산
        
        Returns:
            dict: {
                cpu_id: {
                    'active_pct': float,
                    'frontend_stall_pct': float,
                    'backend_stall_pct': float,
                    'idle_pct': float  (= total stall)
                }
            }
        """
        curr = self.read_counters()
        
        if not self.prev_values:
            self.prev_values = curr
            return None
        
        results = {}
        for cpu in self.cpus:
            prev = self.prev_values[cpu]
            cur = curr[cpu]
            
            cycles = cur['cycles'] - prev['cycles']
            frontend = cur.get('stalled_frontend', 0) - prev.get('stalled_frontend', 0)
            backend = cur.get('stalled_backend', 0) - prev.get('stalled_backend', 0)
            
            if cycles == 0:
                results[cpu] = None
                continue
            
            # Frontend/Backend stall은 overlap 가능
            # Active = cycles에서 stall 제외
            total_stall = frontend + backend
            
            # Stall이 cycles 초과하면 clamp (overlap 때문)
            if total_stall > cycles:
                # 비율로 조정
                ratio = cycles / total_stall
                frontend = int(frontend * ratio)
                backend = int(backend * ratio)
                total_stall = cycles
            
            active = cycles - total_stall
            
            results[cpu] = {
                'active_pct': (active / cycles),
                'frontend_stall_pct': (frontend / cycles),
                'backend_stall_pct': (backend / cycles),
                'idle_pct': (total_stall / cycles),
            }
        
        self.prev_values = curr
        return results
    
    def get_aggregate_utilization(self):
        """전체 CPU 평균 utilization"""
        per_cpu = self.get_utilization()
        if per_cpu is None:
            return None
        
        valid = [v for v in per_cpu.values() if v is not None]
        if not valid:
            return None
        
        return {
            'active_pct': sum(v['active_pct'] for v in valid) / len(valid),
            'frontend_stall_pct': sum(v['frontend_stall_pct'] for v in valid) / len(valid),
            'backend_stall_pct': sum(v['backend_stall_pct'] for v in valid) / len(valid),
            'idle_pct': sum(v['idle_pct'] for v in valid) / len(valid),
        }
    
    def close(self):
        for cpu_fds in self.fds.values():
            for fd in cpu_fds.values():
                os.close(fd)


# ============================================
# 간단한 함수형 인터페이스
# ============================================

_global_monitor = None
_prev_time = None

def init_pmc_monitor(cpus=None):
    """글로벌 모니터 초기화"""
    global _global_monitor
    _global_monitor = OrinPMCMonitor(cpus=cpus)

def get_cpu_active_and_idle():
    """
    CPU active utilization과 idle percentage 반환 (aggregate)
    
    Returns:
        (active_pct, idle_pct) or None if not ready
    """
    global _global_monitor
    if _global_monitor is None:
        init_pmc_monitor()
    
    result = _global_monitor.get_aggregate_utilization()
    if result is None:
        return None
    
    return result['active_pct'], result['idle_pct']

def get_per_core_active_and_idle():
    """
    Per-core active utilization과 idle percentage 반환
    
    Returns:
        dict: {cpu_id: (active_pct, idle_pct)}
    """
    global _global_monitor
    if _global_monitor is None:
        init_pmc_monitor()
    
    result = _global_monitor.get_utilization()
    if result is None:
        return None
    
    return {
        cpu: (v['active_pct'], v['idle_pct']) if v else (None, None)
        for cpu, v in result.items()
    }


# ============================================
# 사용 예시
# ============================================

if __name__ == "__main__":
    print("Jetson Orin NX PMC Monitor")
    print("=" * 60)
    
    # 특정 코어만 모니터링하려면: OrinPMCMonitor(cpus=[0, 1, 2, 3])
    monitor = OrinPMCMonitor()
    
    try:
        while True:
            time.sleep(1)
            
            # Per-core 결과
            per_cpu = monitor.get_utilization()
            if per_cpu is None:
                print("Initializing...")
                continue
            
            # Aggregate
            agg = monitor.get_aggregate_utilization()
            
            print(f"\n[Aggregate] Active: {agg['active_pct']:5.1f}% | "
                  f"Idle(Stall): {agg['idle_pct']:5.1f}% "
                  f"(FE: {agg['frontend_stall_pct']:.1f}%, BE: {agg['backend_stall_pct']:.1f}%)")
            
            # Per-core
            for cpu, data in per_cpu.items():
                if data:
                    print(f"  CPU {cpu}: Active {data['active_pct']:5.1f}% | "
                          f"Stall {data['idle_pct']:5.1f}%")
            
            print("-" * 60)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        monitor.close()