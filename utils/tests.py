import utils

success, output, error = utils.set_cpu_freq(1984000)
if success:
    print("CPU frequency set successfully")
    print(output)
else:
    print(f"Failed: {error}")
    
utils.test_sample()