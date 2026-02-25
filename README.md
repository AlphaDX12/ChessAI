# Chess Training Dashboard & AI Play
## SIMPLE IS GENIUS

A high-performance, refactored PyQt6 GUI for monitoring chess training and playing against the AI. This project adheres to the **"Simple is Genius"** principle, focusing on code conciseness, elegance, and maximum performance.

## Prerequisites

- **Linux** (Tested on Ubuntu/Debian based systems)
- **Python 3.10+**
- **Stockfish** (Chess engine for evaluation and playback)

## Installation

1.  **Install System Dependencies**:
    ```bash
    sudo apt update
    sudo apt install stockfish libxcb-cursor0
    ```
    *Note: `libxcb-cursor0` is often needed for PyQt6 on some Linux distributions.*

2.  **Navigate to the Project**:
    Ensure you are in the project directory:
    ```bash
    cd /home/mathias/2tb/Projects
    ```

3.  **Create a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Launch the Dashboard
To start the main application (Training Monitor + Play Zone):

```bash
python dashboard_gui.py
```

### 2. Features
- **Left Panel (Monitoring)**:
    - **Hardware Vitals**: Real-time CPU, RAM, GPU, and VRAM tracking.
    - **Training Stats**: Monitor samples collected and processing speed.
    - **Start/Stop Training**: Toggle the background training process (`train_az.py`).
    - **Theme**: Instant Dark/Light mode switching.

- **Right Panel (Play Zone)**:
    - **Interactive Board**: Smooth, responsive piece movement.
    - **Play vs AI**: Challenge the latest **MonkeyChessAi Mark2** checkpoint (800 MCTS simulations).
    - **Live Evaluation**: Integrated Stockfish evaluation bar.

- **Bottom Panel (Analytics)**:
    - **Persistent Metrics**: Interactive, multi-color graphs tracking loss and sample stats across sessions.
    - **Log Throttling**: High-performance terminal output designed to prevent UI lag during intense training phases.

## Automated Elo Benchmarking
The suite includes a surgical, high-speed Elo estimation system:
- **Automatic Trigger**: Benchmarking occurs every 250 training episodes.
- **Manual Evaluation**: Toggle estimation with "ESTIMATE ELO" and "RESUME" buttons.
- **Ultra-Fast Matchmaking**: Plays 5 parallel games against Stockfish on core 12 for near-instant strength checks.
- **Visualization**: Live progress and persistent Elo growth curves on the main dashboard.

## Advanced Training Architecture

### 1. Multi-PV Margin Selection
Workers (`mcts_worker.py`) use an advanced policy:
- Stockfish calculates top 3 moves (`Multi-PV = 3`).
- Moves within a 50cp margin are selected with weighted probability, ensuring high-quality diversity.

### 2. Opening Book Support
- Uses `super_gm_book.pgn` to guide the first 8-12 moves (16-24 plies).
- Prevents line memorization and ensures diverse training data.

### 3. Performance Optimizations
- **Geometric Augmentation**: Horizontally flips board positions to double dataset size for free.
- **Zero-Copy Shared Memory**: 300k+ positions passed between processes with zero overhead.
- **Efficient Inference**: `board_to_tensor` optimized for maximum throughput.

## Troubleshooting

- **Stockfish not found**: The app looks for Stockfish at `/usr/games/stockfish`. Ensure it's in your system PATH.
- **PyQt6 xcb-cursor error**: Run `sudo apt install libxcb-cursor0` if the GUI fails to launch.

---
*Optimized for 32GB RAM / 4GB VRAM environments.*

## Advanced Training Features

### 1. Multi-PV Margin Selection
The training workers (`mcts_worker.py`) use an advanced move selection strategy:
- Stockfish calculates the top 3 moves (`Multi-PV = 3`).
- The script identifies all moves within 50 centipawns (0.5 points) of the best move.
- A move is picked randomly from this "Grandmaster approved" set, providing variety without sacrificing quality.

### 2. Opening Book Support
To prevent the AI from playing the same opening lines repeatedly:
- Uses a `super_gm_book.pgn` containing high-level grandmaster games.
- Workers follow random game lines for the first 8-12 moves (16-24 plies).
- This ensures the model trains on thousands of diverse, high-quality opening positions.

## Troubleshooting

- **Stockfish not found**: The app looks for Stockfish at `/usr/games/stockfish` or in your system PATH. If installed elsewhere, ensure it's in your PATH or update `engine_connector.py`.
- **PyQt6 xcb-cursor error**: If the GUI doesn't open, install the missing library: `sudo apt install libxcb-cursor0`.

## 🚀 Elite Features & Robustness (Latest Update)

### 1. Geometric Data Augmentation (2x Speed)
The training pipeline now automatically flips every board position horizontally, effectively **doubling** the dataset size without any extra computational cost. This ensures the bot learns patterns symmetrically for both White and Black.

### 2. Chaos Mode (Human Robustness)
To prevent the AI from overfitting to "perfect" play, we inject controlled chaos into the training data:
- **Blunder Injection (5%)**: The bot is forced to play random moves occasionally to learn how to recover from mistakes.
- **Sub-optimal Paths (20%)**: The bot takes non-optimal but valid paths to see a wider variety of positions.

### 3. Value Jitter & Softmax Policy
- **Value Jitter**: Small random noise is added to evaluation targets to prevent memorization.
- **Softmax Selection**: Moves are selected probabilistically based on their strength (Temperature=50), rather than a hard cutoff.

### 4. Advanced MCTS Search
- **Dirichlet Noise**: Adds creativity to the opening phase.
- **First Play Urgency (FPU)**: Optimizes search exploration for unvisited nodes.
