#!/bin/bash

# --- Dashboard Launch Script ---
# This script ensures the project virtual environment is active 
# and then launches the dashboard.

# Get the directory where the script is located
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "[LAUNCH] Starting Chess Training Dashboard..."

# 1. Check if a virtual environment is already active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[VENV] No virtual environment active. Looking for local venv..."
    
    # 2. Check if the local venv exists
    if [ -d "venv" ]; then
        echo "[VENV] Activating local virtual environment..."
        source venv/bin/activate
    else
        echo "[ERROR] 'venv' directory not found in $PROJECT_DIR."
        echo "Please create a virtual environment first, or run this script from the project root."
        exit 1
    fi
else
    echo "[VENV] Virtual environment already active: $VIRTUAL_ENV"
fi

# 3. Launch the dashboard
echo "[GUI] Launching dashboard_gui.py..."
python3 dashboard_gui.py

# 4. Handle exit
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Dashboard exited with code $EXIT_CODE."
    exit $EXIT_CODE
fi

echo "[DONE] Dashboard closed successfully."
