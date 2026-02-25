# GUI Blueprint: Chess Training Dashboard

The following 5 Python files are the core components of the dashboard and its associated training pipeline:

1.  **`dashboard_gui.py`**: The main entry point. Orchestrates the dual-pane UI, manages the training subprocess, and handles real-time data visualization.
2.  **`hardware_monitor.py`**: A dedicated background thread that monitors system vitals (CPU, RAM, GPU, VRAM) via `psutil` and `pynvml`.
3.  **`engine_connector.py`**: The bridge between the interactive chessboard and the AI. It loads checkpoints for the "Play Zone" and handles move generation.
4.  **`train_az.py`**: The high-performance training loop. Manages shared memory buffers and spawns workers for asynchronous data collection.
5.  **`mcts_worker.py`**: Contains the model architecture (`ChessActorCritic`) and the logic for self-play/data generation used by the training workers.
6.  **`elo_evaluator.py`**: Performance-optimized benchmark tool that estimates Elo by playing 5 parallel games against Stockfish.

### New Dashboard Features:
- **Manual Evaluator Controls**: Buttons to trigger or resume high-precision Elo estimation.
- **Elo Progress Bar**: Real-time feedback on evaluation progress (0-5 games).
- **Elo Trend Selection**: New toggleable metric in the dashboard graph to visualize strength over time.
