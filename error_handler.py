# SIMPLE IS GENIUS
import sys, warnings # System-specific parameters and functions
# Suppress the pynvml deprecation warning to keep the console clean
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import traceback # Extraction and formatting of stack traces
import platform # Access to underlying platform's identifying data
import os # Miscellaneous operating system interfaces

# SAFE DEPENDENCY LOADING
try: import psutil 
except ImportError: psutil = None
try: import pynvml 
except ImportError: pynvml = None

class ErrorHandler:
    """Diagnostic suite for structured crash reporting and environment diagnostics."""
    
    @staticmethod
    def initialize():
        """Installs the error handler as the global exception hook."""
        sys.excepthook = ErrorHandler.handle_exception
        # For PyQt6 compatibility: Ensure we catch exceptions in the event loop
        try:
            from PyQt6.QtCore import pyqtRemoveInputHook
            pyqtRemoveInputHook() # Often necessary for debuggers/custom hooks
        except ImportError: pass
        print("[SUCCESS] Error Handler Active")

    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Intercepts unhandled exceptions and produces a comprehensive terminal report."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Standard behavior for CTRL+C: just pass through
            return

        # 1. Visual Header
        print("\n" + "="*80)
        print(" [CRITICAL SYSTEM FAILURE DETECTED] ")
        print("="*80)

        # 2. Crash Summary
        print(f"\n[ERROR TYPE]: {exc_type.__name__}")
        print(f"[DESCRIPTION]: {exc_value}")

        # 3. Structured Traceback
        print("\n" + "-"*40)
        print(" [TRACEBACK ANALYSIS] ")
        print("-"*40)
        
        tb = traceback.extract_tb(exc_traceback)
        for i, frame in enumerate(tb):
            # Indent based on depth to show 'flow'
            indent = "  " * i
            print(f"{indent}File: {os.path.basename(frame.filename)}")
            print(f"{indent}   Line: {frame.lineno} | Function: {frame.name}")
            print(f"{indent}   Code: {frame.line}")
            print()

        # 4. Environment Context
        print("-"*40)
        print(" [ENVIRONMENT DIAGNOSTICS] ")
        print("-"*40)
        print(f"OS/Kernel: {platform.system()} {platform.release()}")
        print(f"Python:    {sys.version.split()[0]}")
        
        if psutil:
            print(f"CPU Load:  {psutil.cpu_percent()}%")
            print(f"RAM Avail: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        else:
            print("System Vitals: 'psutil' missing")
        
        if pynvml:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                print(f"GPU:       {name} ({temp} C)")
                pynvml.nvmlShutdown()
            except: print("GPU:       Driver issue or not found")
        else: print("GPU:       Monitoring library missing")

        # 5. Suggestions
        print("\n" + "-"*40)
        print(" [SUGGESTIONS] ")
        print("-"*40)
        
        error_msg = str(exc_value).lower()
        if issubclass(exc_type, ImportError):
            module_name = str(exc_value).split("'")[-2] if "'" in str(exc_value) else "module"
            print(f"-> Dependency missing. Try: ./venv/bin/pip install {module_name}")
        elif issubclass(exc_type, TypeError) and "sharedreader" in error_msg:
            print("-> SharedReader architectural conflict detected.")
            print("-> Note: Training metrics are now streamed directly via logs.")
            print("-> Recommendation: Run a 'git pull' or verify you have the latest code.")
        elif "cuda" in error_msg:
            print("-> GPU error detected. Verify your NVIDIA drivers and CUDA installation.")
        else:
            print("-> Review the local modifications in the files listed above.")
            print("-> Check for unclosed processes or shared memory locks.")

        print("\n" + "="*80)
        print(" [TERMINATING PROCESS] ")
        print("="*80 + "\n")
        
        # Finally, terminate the app
        sys.exit(1)

# Auto-initialize on import if desired, but we'll call it explicitly for clarity.
