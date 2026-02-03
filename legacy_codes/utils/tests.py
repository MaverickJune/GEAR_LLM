try:
    # 패키지로 실행될 때
    from . import utils
    from . import read_power
except ImportError:
    # 직접 스크립트로 실행될 때
    import utils
    import read_power
import time

success, output, error = utils.set_cpu_freq(1984000)
if success:
    print("CPU frequency set successfully")
    print(output)
else:
    print(f"Failed: {error}")

def get_core_time(num_cpu):
    with open('/proc/stat') as f: 
        lines = f.readlines()
    idles,totals = [0]*num_cpu, [0]*num_cpu
    for i, l in enumerate(lines[1:num_cpu+1]):
        fields = [float(column) for column in l.strip().split()[1:]]
        idles[i] = fields[3]
        totals[i] = sum(fields)
    return idles, totals

idles, totals = get_core_time(8)
print("CPU Idle Times:", idles)
print("CPU Total Times:", totals)