import utils
import read_power

success, output, error = utils.set_cpu_freq(1984000)
if success:
    print("CPU frequency set successfully")
    print(output)
else:
    print(f"Failed: {error}")
    
utils.test_sample()
p_dict = read_power.read_hwmon_power_w()
print(p_dict)