# SIMPLE IS GENIUS
import sys, os, signal, atexit # Process and signal handling
from error_handler import ErrorHandler # Crash diagnostics and reporting
ErrorHandler.initialize() # Install global exception hook immediately

# Ensure project root is in the path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Single-Instance Enforcement ---
_LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dashboard.lock")

def _is_process_running(pid):
    """Returns True if a process with the given PID is currently running."""
    try:
        os.kill(pid, 0) # Signal 0 = existence check, no actual signal sent
        return True
    except (OSError, ProcessLookupError):
        return False

def _acquire_instance_lock():
    """Tries to acquire the single-instance lock. Returns True if successful."""
    if os.path.exists(_LOCK_FILE):
        try:
            with open(_LOCK_FILE) as f:
                existing_pid = int(f.read().strip())
            if _is_process_running(existing_pid):
                return False  # Another real instance is running
        except (ValueError, IOError):
            pass  # Stale/corrupt lockfile — overwrite it
    # Write our PID
    with open(_LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(_LOCK_FILE) and os.remove(_LOCK_FILE))
    return True

from PyQt6.QtWidgets import QApplication, QMessageBox # Main application wrapper
from PyQt6.QtCore import QTimer # Periodic timer for UI event loop forcing
from gui.dashboard import Dashboard # The main UI orchestrator

if __name__ == "__main__":
    # 1. Initialize the Qt Application context
    app = QApplication(sys.argv)

    # 2. Single-instance check — show error and exit if dashboard is already running
    if not _acquire_instance_lock():
        msg = QMessageBox()
        msg.setWindowTitle("Chess Training Dashboard")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText("Dashboard is already running.\n\nOnly one instance can be open at a time to prevent data conflicts.")
        msg.exec()
        sys.exit(0)

    # 3. Spawn the primary Dashboard controller
    try:
        window = Dashboard()
    except Exception:
        # Pass to the global exception handler
        ErrorHandler.handle_exception(*sys.exc_info())
        sys.exit(1)

    # 4. Handle CTRL+C gracefully by linking SIGINT to window closure
    signal.signal(signal.SIGINT, lambda *_: (print("\n[STOP] SIGINT received."), window.close()))
    
    # 5. Enforce Python signal handling in the Qt event loop
    timer = QTimer()
    timer.start(500) # Poll for signals every 500ms
    timer.timeout.connect(lambda: None) 
    
    # 6. Launch the visual window and enter the main event loop
    window.show()
    sys.exit(app.exec())

