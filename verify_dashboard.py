# SIMPLE IS GENIUS
import sys, os # Core system utilities

# Add project root to path to ensure local imports Resolve correctly
sys.path.append(os.getcwd())

try:
    print("Attempting project-wide imports...")
    # Attempt to load the main dashboard to verify environment readiness
    import dashboard_gui
    print("[DONE] All core modules Resolve successfully.")
except ImportError as e:
    # Diagnostic catch for missing dependencies (venv issues)
    print(f"[ERROR] ImportError: {e}")
    sys.exit(1)
except Exception as e:
    # Catch-all for unexpected initialization failures
    print(f"[CRITICAL] Critical error during import: {e}")
    sys.exit(1)

print("Verification script finished: READY FOR STABLE OPERATION.")
