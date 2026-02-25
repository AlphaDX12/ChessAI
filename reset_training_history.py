#!/usr/bin/env python3
"""
Training History Reset Utility
Backs up and resets all training history files for a fresh start with new model architecture.
"""

import os  # File system operations
import json  # JSON file handling
import shutil  # High-level file operations like copying directories
from datetime import datetime  # Timestamp generation for backup folders

def reset_training_history():
    """Main function to backup old history and create fresh files."""
    
    # Define the checkpoint directory where all history files are stored
    checkpoint_dir = "Checkpoint"
    
    # Create timestamp for unique backup folder name (format: backup_2026-02-17_195937)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_dir = os.path.join(checkpoint_dir, f"backup_{timestamp}")
    
    # List of history files that need to be reset for the new model architecture
    history_files = [
        "training_history.json",  # Training metrics (loss, samples, etc.)
        "elo_history.json",       # Elo rating progression over time
        "elo_eval_state.json"     # Current Elo evaluation state
    ]
    
    print("=" * 60)  # Visual separator for console output
    print("Training History Reset Utility")
    print("=" * 60)
    
    # Check if checkpoint directory exists before proceeding
    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] Error: {checkpoint_dir} directory not found!")
        print("   Please run this script from the project root directory.")
        return
    
    # Create backup directory to preserve old data
    print(f"\n[BACKUP] Creating backup directory: {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)  # exist_ok prevents error if dir already exists
    
    # Track how many files were actually backed up
    backed_up_count = 0
    
    # Iterate through each history file and back it up if it exists
    for filename in history_files:
        source_path = os.path.join(checkpoint_dir, filename)  # Full path to original file
        
        # Only backup files that actually exist (some might not be created yet)
        if os.path.exists(source_path):
            dest_path = os.path.join(backup_dir, filename)  # Full path to backup location
            shutil.copy2(source_path, dest_path)  # copy2 preserves metadata (timestamps, etc.)
            print(f"   [DONE] Backed up: {filename}")
            backed_up_count += 1
        else:
            print(f"   [SKIP] Skipped: {filename} (doesn't exist)")
    
    print(f"\n[SAVE] Backed up {backed_up_count} file(s) to {backup_dir}")
    
    # Now create fresh, empty history files for the new model architecture
    print("\n[RESET] Creating fresh history files...")
    
    # Create empty training history with proper JSON structure
    training_history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(training_history_path, 'w') as f:
        # Initialize with empty arrays for each metric, starting from episode 0
        json.dump({
            "ep": [],        # Episode numbers
            "loss": [],      # Total loss values
            "v_loss": [],    # Value head loss
            "p_loss": [],    # Policy head loss  
            "samples": [],   # Total training samples processed
            "elo": []        # Skill estimate history
        }, f, indent=2)  # indent=2 makes the JSON human-readable
    print(f"   [DONE] Created: training_history.json")
    
    # Create empty Elo history with proper JSON structure
    elo_history_path = os.path.join(checkpoint_dir, "elo_history.json")
    with open(elo_history_path, 'w') as f:
        # Initialize with empty arrays for Elo ratings over time
        json.dump({
            "ep": [],    # Episode numbers when Elo was evaluated
            "elo": []    # Corresponding Elo ratings
        }, f, indent=2)
    print(f"   [DONE] Created: elo_history.json")
    
    # Create fresh Elo evaluation state file
    elo_state_path = os.path.join(checkpoint_dir, "elo_eval_state.json")
    with open(elo_state_path, 'w') as f:
        # Initialize with default state (no evaluation in progress)
        json.dump({
            "running": False,  # Whether an Elo evaluation is currently running
            "episode": 0       # Last episode number evaluated
        }, f, indent=2)
    print(f"   [DONE] Created: elo_eval_state.json")
    
    # Success message with instructions
    print("\n" + "=" * 60)
    print("[DONE] History reset complete!")
    print("=" * 60)
    print("\nYour training graphs will now start from episode 0.")
    print("Old data has been safely backed up in:")
    print(f"   {backup_dir}")
    print("\nYou can now start the dashboard with fresh history.")
    print("=" * 60)

# Standard Python idiom: only run if script is executed directly (not imported)
if __name__ == "__main__":
    reset_training_history()  # Execute the main reset function
