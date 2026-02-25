# SIMPLE IS GENIUS
import time, psutil, warnings # Standard libraries for timing and system monitoring
# Suppress the pynvml deprecation warning to keep the console clean during training
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

try: 
    import pynvml
except ImportError:
    pynvml = None

from PyQt6.QtCore import QThread, pyqtSignal # Qt multi-threading components

class HardwareWorker(QThread): # Background polling thread for system metrics
    """Background thread that polls system load without blocking the main UI."""
    # Custom signal that emits a dictionary of metrics every second to the UI
    stats_signal = pyqtSignal(dict) # Data bridge to the Dashboard HUD

    def __init__(self, parent=None): # Constructor for the hardware monitor
        super().__init__(parent) # Initialize base QThread
        self.running = True # Master control flag for the while loop
        self.nvml_initialized = False # Tracks if GPU telemetry is actually active
        self.gpu_history = [] # Circular buffer to smooth out utilization spikes
        self.total_vram_gb = 0 # Detected total VRAM capacity
        
        # Initialize the NVIDIA Management Library (NVML) for GPU access
        if pynvml: 
            try: 
                pynvml.nvmlInit() 
                self.nvml_initialized = True 
                # Initial detection of VRAM capacity
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.total_vram_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3
                print(f"[DONE] GPU Monitoring Active: {pynvml.nvmlDeviceGetName(handle)} ({self.total_vram_gb:.1f}GB VRAM)")
            except Exception as e: 
                print(f"[WARNING] GPU Monitoring not available: {e}")

    def _safe_nvml(self, func, handle, *args, default=0):
        """Standardizes error handling for optional GPU metrics."""
        try:
            return func(handle, *args)
        except Exception:
            return default

    def run(self): # Main execution loop (executed in dedicated thread)
        """Infinite loop that performs high-frequency polling of CPU and GPU loads."""
        while self.running: # Continue until stop() is called
            # 1. Fetch CPU and RAM metrics
            stats = {
                'cpu_percent': psutil.cpu_percent(), # Average total system load
                'cpu_cores': psutil.cpu_percent(percpu=True), # Granular per-core breakdown
                'ram_gb': psutil.virtual_memory().used / 1024**3 # Active system RAM
            }
            
            # 2. Add detailed GPU metrics if NVML is responding
            if self.nvml_initialized:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # Utilization (may return None if failing, but we handle it)
                    util = self._safe_nvml(pynvml.nvmlDeviceGetUtilizationRates, handle, default=None)
                    gpu_val = util.gpu if util else 0
                    mem_val = util.memory if util else 0

                    # Use a 5-point moving average to smooth the master utilization rate
                    self.gpu_history = (self.gpu_history + [gpu_val])[-5:]
                    avg_util = int(sum(self.gpu_history) / len(self.gpu_history))
                    
                    # Memory info
                    mem_info = self._safe_nvml(pynvml.nvmlDeviceGetMemoryInfo, handle, default=None)
                    vram_used = (mem_info.used / 1024**3) if mem_info else 0

                    # Package rich NVIDIA telemetry with per-call safety
                    stats.update({
                        'gpu_util': avg_util, # Smoothed Graphics Load
                        'gpu_mem_util': mem_val, # Dedicated Memory Controller Load
                        'vram_gb': vram_used,
                        'vram_total': self.total_vram_gb,
                        'gpu_clock': self._safe_nvml(pynvml.nvmlDeviceGetClockInfo, handle, 0, default=0),
                        'gpu_pwr': self._safe_nvml(pynvml.nvmlDeviceGetPowerUsage, handle, default=0) / 1000.0,
                        'gpu_temp': self._safe_nvml(pynvml.nvmlDeviceGetTemperature, handle, 0, default=0),
                        'gpu_fan': self._safe_nvml(pynvml.nvmlDeviceGetFanSpeed, handle, default=0)
                    })
                except Exception as e:
                    stats['gpu_error'] = str(e)
            
            # 3. Broadcast snapshot to UI
            self.stats_signal.emit(stats) 
            time.sleep(1) # Keep polling overhead to near-zero

    def stop(self): # Halts the monitor thread cleanly
        """Gracefully halts the monitoring loop and releases driver locks."""
        print("[STOP] [HardwareWorker] Stopping monitoring thread...")
        self.running = False # Signal the while loop to break
        if not self.wait(2000): # Block until the thread fully terminates (2s timeout)
            print("[WARNING] [HardwareWorker] Thread hang detected, forcing termination.")
            self.terminate()
            self.wait()
        if self.nvml_initialized: # If drivers were open
            try: 
                pynvml.nvmlShutdown() # Cleanly release NVML to prevent driver memory leaks
                print("[DONE] [HardwareWorker] NVML cleanup complete.")
            except Exception as e: 
                print(f"[WARNING] [HardwareWorker] NVML cleanup failed: {e}")
