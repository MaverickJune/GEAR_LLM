#!/usr/bin/env python3
"""
Test script to verify DVFS setup before running full training
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from utils.utils import set_cpu_freq, sample_pmu_only
        print("  ✓ utils.utils")
    except Exception as e:
        print(f"  ✗ utils.utils: {e}")
        return False
    
    try:
        from utils.monitor import Monitor
        print("  ✓ utils.monitor")
    except Exception as e:
        print(f"  ✗ utils.monitor: {e}")
        return False
    
    try:
        from utils.api_client import DVFSClient
        print("  ✓ utils.api_client")
    except Exception as e:
        print(f"  ✗ utils.api_client: {e}")
        return False
    
    try:
        import PyPerf as Perf
        print("  ✓ PyPerf (perf_lib)")
    except Exception as e:
        print(f"  ✗ PyPerf: {e}")
        print("    Run: cd perf_lib && bash build_perf.sh")
        return False
    
    return True


def test_cpu_freq():
    """Test CPU frequency control"""
    print("\nTesting CPU frequency control...")
    
    from utils.utils import set_cpu_freq
    
    # Try setting a safe middle frequency
    test_freq = 1420800  # 1.42 GHz
    
    print(f"  Attempting to set CPU freq to {test_freq} kHz...")
    success, stdout, stderr = set_cpu_freq(test_freq)
    
    if success:
        print("  ✓ CPU frequency control works")
        return True
    else:
        print(f"  ✗ CPU frequency control failed: {stderr}")
        print("    Run: sudo chmod 666 /sys/devices/system/cpu/cpu*/cpufreq/*")
        return False


def test_monitor():
    """Test power/temperature monitoring"""
    print("\nTesting monitor...")
    
    from utils.monitor import Monitor
    import time
    
    monitor = Monitor()
    
    print("  Querying sensors...")
    try:
        timestamp, power_data, temp_data = monitor.query()
        
        print(f"  ✓ Monitor working")
        print(f"    Timestamp: {timestamp:.3f}")
        print(f"    Power rails: {len(power_data)}")
        if power_data:
            for key, value in list(power_data.items())[:3]:
                print(f"      {key}: {value:.2f} mW")
        print(f"    Temp sensors: {len(temp_data)}")
        if temp_data:
            for key, value in list(temp_data.items())[:3]:
                print(f"      {key}: {value:.2f} °C")
        
        return True
    except Exception as e:
        print(f"  ✗ Monitor failed: {e}")
        return False


def test_pmu():
    """Test PMU counter sampling"""
    print("\nTesting PMU counters...")
    
    try:
        from utils.utils import sample_pmu_only
        
        cpu = [0, 1, 2, 3]
        events = [0]  # cycles
        
        print("  Sampling PMU counters...")
        pmu_data = sample_pmu_only(events, cpu, 100000)  # 100ms
        
        print(f"  ✓ PMU sampling works")
        print(f"    Counters collected: {len(pmu_data)}")
        for key, value in list(pmu_data.items())[:4]:
            print(f"      {key}: {value}")
        
        return True
    except Exception as e:
        print(f"  ✗ PMU sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_client(server_url):
    """Test API client connection to server"""
    print(f"\nTesting API client connection to {server_url}...")
    
    from utils.api_client import DVFSClient
    
    client = DVFSClient(server_url=server_url)
    
    print("  Attempting to initialize model...")
    if client.init_model(model_type="dqn_nx"):
        print(f"  ✓ Successfully connected and initialized model")
        print(f"    Model ID: {client.model_id}")
        
        # Test getting action
        print("  Testing action request...")
        test_state = {
            'timestamp': 0.0,
            'power': {'rail1': 5000.0},
            'temp': {'sensor1': 50.0},
            'pmu': {'cpu0_cycles': 1000000}
        }
        
        response = client.get_action(test_state, train=False)
        if response:
            print(f"  ✓ Received action: {response['action']}")
        else:
            print(f"  ✗ Failed to get action")
            return False
        
        # Cleanup
        client.remove_model()
        return True
    else:
        print(f"  ✗ Failed to connect to server")
        print(f"    Make sure server is running: python Perf-Trainer/app.py")
        return False


def test_llm_generation():
    """Test LLM generation (optional - requires model)"""
    print("\nTesting LLM generation...")
    
    model_path = "/home/nxc/wjbang/models/Llama-3.2-1B-Instruct-f16.gguf"
    
    if not os.path.exists(model_path):
        print(f"  ⚠ Model not found: {model_path}")
        print(f"    Skipping LLM test")
        return True  # Not a failure, just skip
    
    try:
        from pathlib import Path
        llama_path = "/home/nxc/wjbang/llama.cpp"
        sys.path.insert(0, str(Path(llama_path)))
        from gear_decode.gear_generate import GearGenerator
        
        lib_path = "/home/nxc/wjbang/llama.cpp/build/lib/libgear_decode.so"
        
        if not os.path.exists(lib_path):
            print(f"  ⚠ Library not found: {lib_path}")
            print(f"    Skipping LLM test")
            return True
        
        print("  Initializing generator...")
        generator = GearGenerator(lib_path=lib_path)
        
        print("  Running quick generation test (5 tokens)...")
        result = generator.generate(
            model_path=model_path,
            prompt="Hello",
            n_predict=5,
            use_instruct=False,
            n_threads=2,
            enable_flash_attn=False
        )
        
        if result.is_success:
            print(f"  ✓ LLM generation works")
            print(f"    Generated: {result.output_text}")
            print(f"    Tokens/sec: {result.tokens_per_second:.2f}")
            return True
        else:
            print(f"  ✗ LLM generation failed")
            return False
            
    except Exception as e:
        print(f"  ✗ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("DVFS Setup Verification")
    print("="*80)
    
    results = []
    
    # Basic tests
    results.append(("Imports", test_imports()))
    results.append(("CPU Frequency", test_cpu_freq()))
    results.append(("Monitor", test_monitor()))
    results.append(("PMU Counters", test_pmu()))
    
    # Optional: API test (requires server)
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
        results.append(("API Client", test_api_client(server_url)))
    else:
        print("\n⚠ Skipping API test (no server URL provided)")
        print("  Run: python test_dvfs_setup.py http://SERVER_IP:5000")
    
    # Optional: LLM test
    results.append(("LLM Generation", test_llm_generation()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to run nx_client.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before running.")
        return 1


if __name__ == "__main__":
    exit(main())
