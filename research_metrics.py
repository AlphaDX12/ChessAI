# SIMPLE IS GENIUS
import pynvml
import psutil
import time

def research():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Initialize CPU polling
    psutil.cpu_percent(percpu=True)
    time.sleep(1)

    print("--- GPU Engines ---")
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU Util: {util.gpu}%")
        print(f"Memory Util: {util.memory}%")
    except Exception as e: print(f"GPU Engine Error: {e}")

    print("\n--- Clocks ---")
    try:
        print(f"Graphics Clock: {pynvml.nvmlDeviceGetClockInfo(handle, 0)} MHz")
        print(f"SM Clock: {pynvml.nvmlDeviceGetClockInfo(handle, 1)} MHz")
        print(f"Mem Clock: {pynvml.nvmlDeviceGetClockInfo(handle, 2)} MHz")
    except Exception as e: print(f"Clock Error: {e}")

    print("\n--- Power ---")
    try:
        print(f"Power Usage: {pynvml.nvmlDeviceGetPowerUsage(handle) / 1000:.2f} W")
        print(f"Power Limit: {pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000:.2f} W")
    except Exception as e: print(f"Power Error: {e}")

    print("\n--- CPU Cores ---")
    cores = psutil.cpu_percent(percpu=True)
    print(f"Per-core usage ({len(cores)} cores): {cores}")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    research()
