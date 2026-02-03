from GearLLM.jetson_utils.state_utils import OrinNaiveStateMonitor

monitor = OrinNaiveStateMonitor()
state = monitor.get_state()
print("State vector:", state)