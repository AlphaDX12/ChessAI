# SIMPLE IS GENIUS
import warnings # Control diagnostic output
# Suppress the pynvml deprecation warning used by torch.cuda and others
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import torch # Deep learning framework for model training
import torch.nn as nn # Neural network components
import torch.optim as optim # Optimization algorithms (Adam, SGD, etc.)
import numpy as np # Numerical operations and array handling
import multiprocessing as mp # Parallel process management for workers
import os, time, signal, glob # System, timing, signal handling, and file patterns
import torch.nn.functional as F # High-performance functional activations
from chess_engine import ChessActorCritic, OBS_SHAPE, MOVE_ACTION_DIM # Core AI definitions
from chess_engine import run_game_worker, NUM_RESIDUAL_BLOCKS # Simulation worker and config
from error_handler import ErrorHandler # Crash diagnostics and reporting

def handle_sigterm(signum, frame):
    """Bridge SIGTERM to KeyboardInterrupt for clean GUI-triggered shutdowns."""
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, handle_sigterm)

# --- Training Hyper-Parameters ---
BATCH_SIZE = 1024 # Reduced to 1024 for stability on 4GB VRAM
LEARNING_RATE = 1e-4 # Step size for the optimizer
WEIGHT_DECAY = 1e-4 # L2 regularization to prevent overfitting
# Use absolute path for Checkpoints to ensure it works regardless of GUI launch directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Checkpoint") 
MAX_SAMPLES = 250_000 # [ADJUSTED] Set back to 250k as requested to fit in available RAM
TRAIN_INTERVAL_SAMPLES = 5000 # Train every time we get this many new samples
STEPS_PER_INTERVAL = 32 # Reduced from 128 to significantly reduce sample reuse ratio for stability
WARMUP_EPISODES = 100 # Linear LR warmup phase

def _kill_worker_tree(pid):
    """Recursively kills a process and all its children to prevent 'zombie' engines."""
    try:
        import subprocess # Required for shell commands
        # Use pgrep to find all children of the given PID
        children = subprocess.check_output(["pgrep", "-P", str(pid)]).decode().split()
        for child_pid in children: _kill_worker_tree(int(child_pid)) # Recursive call for sub-children
        os.kill(pid, signal.SIGKILL) # Force-kill the target process
    except: pass # Ignore errors if process already exited

def train():
    """Main training orchestrator: Manages workers, memory, and the training loop."""
    ErrorHandler.initialize() # Enable crash reporting
    print("Initializing Chess Training System...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Ensure checkpoint folder exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    print(f"Training on: {device}")

    # --- Shared Memory Initialization ---
    # We use shared arrays so workers can write data directly into the main training buffer.
    # 'H' = uint16 (2 bytes). Interpreted as float16 via NumPy for 50% RAM savings.
    s_arr = mp.RawArray('H', MAX_SAMPLES * 20 * 8 * 8) # Board state tensor storage (20 planes)
    p_arr = mp.RawArray('H', MAX_SAMPLES * MOVE_ACTION_DIM) # Policy probability storage
    v_arr = mp.RawArray('H', MAX_SAMPLES) # Value (win/loss) storage
    q_arr = mp.RawArray('B', MAX_SAMPLES) # Quality (categorical) storage
    ptr = mp.Value('i', 0) # Head pointer for the circular buffer
    count = mp.Value('i', 0) # Total samples currently available for training
    total_samples = mp.Value('L', 0) # Rolling lifetime sample counter
    diversity_samples = mp.Value('L', 0) # Counter for high-entropy exploration moves
    lock = mp.Lock() # Mutex to prevent race conditions during pointer updates

    # Convert shared RawArrays to NumPy views for ultra-fast slicing during training
    # These MUST be defined before we attempt to load a saved buffer from disk
    states = np.frombuffer(s_arr, dtype=np.float16).reshape(MAX_SAMPLES, 20, 8, 8)
    policies = np.frombuffer(p_arr, dtype=np.float16).reshape(MAX_SAMPLES, MOVE_ACTION_DIM)
    values = np.frombuffer(v_arr, dtype=np.float16).reshape(MAX_SAMPLES, 1)
    qualities = np.frombuffer(q_arr, dtype=np.uint8).reshape(MAX_SAMPLES, 1)

    # --- Model & Optimizer Setup ---
    model = ChessActorCritic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # --- Training Enhancements ---
    scaler = torch.amp.GradScaler('cuda') # For Mixed Precision (AMP)
    
    # Combined Scheduler: Linear Warmup + SMOOTH Cosine Decay (Cycle every 500k samples)
    warmup_sch = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPISODES)
    # T_max is 250,000 samples / TRAIN_INTERVAL_SAMPLES (5,000) = 50 intervals -> Increased to 1000 for stability
    cosine_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[WARMUP_EPISODES])

    # --- Checkpoint Resumption ---
    # We follow a multi-track resume strategy:
    # 1. training_state.pth: Latest safe-exit checkpoint (Full State)
    # 2. full_state_ep_*.pth: Periodic full-state checkpoints (Full State)
    # 3. model_ep_*.pth: Periodic model-only checkpoints (Weights Only)
    
    import re
    def get_episode_from_path(path):
        match = re.search(r'ep_(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0

    start_ep = 0 
    state_path = f"{CHECKPOINT_DIR}/training_state.pth"
    full_ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/full_state_ep_*.pth"), key=get_episode_from_path, reverse=True)
    model_ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/model_ep_*.pth"), key=get_episode_from_path, reverse=True)

    candidates = []
    if os.path.exists(state_path):
        candidates.append((state_path, "FULL"))
    for ckpt in full_ckpts:
        candidates.append((ckpt, "FULL"))
    for ckpt in model_ckpts:
        candidates.append((ckpt, "WEIGHTS"))

    resumed = False
    for resume_path, resume_type in candidates:
        print(f"Attempting to restore {resume_type} training state from {resume_path}...")
        try:
            state = torch.load(resume_path, map_location=device, weights_only=False)
            
            # Common model loading logic
            checkpoint_state = state['model'] if isinstance(state, dict) and 'model' in state else state
            new_state = {k.replace('_orig_mod.', ''): v for k, v in checkpoint_state.items()}
            model.load_state_dict(new_state)
            
            if resume_type == "FULL":
                optimizer.load_state_dict(state['optimizer'])
                start_ep = state['episode'] 
                ptr.value, count.value = state['ptr'], state['count']
                total_samples.value, diversity_samples.value = state['total_samples'], state['diversity_samples']
                # Restore buffer contents
                np.copyto(states[:count.value].reshape(-1), state['buffer_s'].reshape(-1))
                np.copyto(policies[:count.value].reshape(-1), state['buffer_p'].reshape(-1))
                np.copyto(values[:count.value].reshape(-1), state['buffer_v'].reshape(-1))
                if 'buffer_q' in state:
                    np.copyto(qualities[:count.value].reshape(-1), state['buffer_q'].reshape(-1))
                print(f"Resumed from Episode {start_ep} with {count.value} buffer samples.")
            else:
                start_ep = state.get('episode', 0) if isinstance(state, dict) else 0
                print(f"Resumed weights only from {resume_path} (Episode {start_ep}).")
            
            resumed = True
            torch.cuda.empty_cache()
            break # Success!
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint {resume_path}: {e}")
            continue

    if not resumed:
        print("No valid checkpoints found. Starting with fresh weights.")

    # [Optimization] Compile the model AFTER loading for maximum compatibility and speed
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model for high-speed training...")
        model = torch.compile(model) 

    # Loss Functions: MSE for board evaluation (value)
    # Note: Policy loss is computed manually via soft-target cross-entropy in the loop
    v_loss_fn = nn.MSELoss() 

    # --- Worker Launch ---
    num_workers = max(1, mp.cpu_count() - 2)
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=run_game_worker, args=(i, None, s_arr, p_arr, v_arr, q_arr, ptr, count, total_samples, diversity_samples, lock))
        p.start()
        workers.append(p)
    print(f"Launched {num_workers} game generation workers.")

    def save_state(path, full=True):
        """Helper to save training state atomically. If full=True, saves the replay buffer too."""
        # Snapshot count under lock to avoid data race with workers
        with lock:
            snap_count = count.value
            snap_ptr = ptr.value
            
        msg = f"[SAVE] Saving FULL State..." if full else "[SAVE] Saving Model weights..."
        print(f"{msg} ({snap_count:,} samples) to {os.path.basename(path)}...", end="", flush=True)
        
        data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'episode': episode,
            'ptr': snap_ptr,
            'count': snap_count,
            'total_samples': total_samples.value,
            'diversity_samples': diversity_samples.value
        }
        if full:
            data['buffer_s'] = states[:snap_count].copy()
            data['buffer_p'] = policies[:snap_count].copy()
            data['buffer_v'] = values[:snap_count].copy()
            data['buffer_q'] = qualities[:snap_count].copy()
        
        # Atomic Save: write to .tmp and rename
        temp_path = path + ".tmp"
        try:
            torch.save(data, temp_path)
            os.replace(temp_path, path)
            print(" DONE [DONE]")
        except Exception as e:
            print(f" FAILED: {e}")
            if os.path.exists(temp_path): os.remove(temp_path)

    # --- Main Training Loop ---
    last_train_count = total_samples.value 
    last_gen_time = time.time()
    episode = start_ep 
    
    try:
        while True:
            # Check if enough new samples have been generated by workers
            current_total = total_samples.value
            new_samples = current_total - last_train_count
            
            if new_samples < TRAIN_INTERVAL_SAMPLES:
                time.sleep(1) 
                continue
            
            # Guard: ensure we have enough samples to form at least one full batch
            if count.value < BATCH_SIZE:
                time.sleep(1)
                continue

            # Calculate Generation Speed (GenSPS)
            gen_duration = time.time() - last_gen_time
            gen_speed = (current_total - last_train_count) / max(0.001, gen_duration)
            
            last_train_count = current_total # Reset the trigger
            last_gen_time = time.time() # Reset generation timer
            episode += 1 # Increment session counter
            
            # Snap current buffer size and head pointer under lock for consistent indexing
            with lock:
                current_count = count.value 
                current_ptr = ptr.value
            
            # Perform high-intensity training steps
            model.train() 
            train_start = time.time()
            total_loss_val, total_p_loss, total_v_loss = 0, 0, 0
            
            # --- Recency-Biased Sampling Weights ---
            # Linear weights where most recent samples have highest probability
            raw_weights = np.arange(1, current_count + 1, dtype=np.float64)
            # Circular buffer reordering: map weight N to the sample at (ptr-1), weight N-1 to (ptr-2)...
            # Simplest approach: roll weights so the highest weight aligns with the 'latest' index
            # Index ptr-1 is the newest, index ptr is the oldest.
            weights = np.roll(raw_weights, current_ptr) 
            probs = weights / weights.sum()
            
            # [Intelligence/Stability] Scale intensity by buffer fill ratio
            # Prevents over-training on a tiny dataset during the first few episodes.
            fill_ratio = current_count / MAX_SAMPLES
            dynamic_steps = max(64, int(STEPS_PER_INTERVAL * fill_ratio))
            
            # [Optimization] replace=True is ~100x faster for large n and has zero impact on learning quality.
            all_indices = [np.random.choice(current_count, BATCH_SIZE, replace=True, p=probs) for _ in range(dynamic_steps)]
            
            for step_i in range(dynamic_steps):
                # Use pre-sampled indices for this step
                indices = all_indices[step_i]
                
                # [Performance] Combine dtype casting and device transfer in a single operation
                # torch.as_tensor() avoids unnecessary copies from NumPy.
                batch_s = torch.as_tensor(states[indices]).to(device, dtype=torch.float32, non_blocking=True)
                batch_p = torch.as_tensor(policies[indices]).to(device, dtype=torch.float32, non_blocking=True)
                batch_v = torch.as_tensor(values[indices]).to(device, dtype=torch.float32, non_blocking=True)
                batch_q = torch.as_tensor(qualities[indices]).to(device, dtype=torch.long, non_blocking=True).view(-1)

                optimizer.zero_grad(set_to_none=True)
                
                # Automatic Mixed Precision (AMP) Forward Pass
                with torch.amp.autocast('cuda'):
                    pred_p, pred_v, pred_q = model(batch_s)
                    
                    # Label Smoothing (ε=0.01) to handle sparse Stockfish targets
                    eps = 0.01
                    smoothed_p = batch_p * (1.0 - eps) + (eps / MOVE_ACTION_DIM)
                    
                    # 1. Policy loss: Move Ranking distillation
                    log_probs = F.log_softmax(pred_p, dim=1)
                    p_loss = -(smoothed_p * log_probs).sum(dim=1).mean()
                    
                    # Entropy Regularization
                    probs_ent = log_probs.exp()
                    entropy = -(probs_ent * log_probs).sum(dim=1).mean()
                    
                    # 2. Value loss: Evaluation Bar alignment
                    # CP-to-Value scaling: Stockfish centipawn scores are scaled to [-1, 1] using tanh(score / 300)
                    # This is already handled by the data generation, so batch_v is already in the correct range.
                    v_loss = v_loss_fn(pred_v, batch_v) # Normalized weighting
                    
                    # 3. Quality loss: Categorical move assessment
                    # [Weighted Loss] We penalize Blunders (class 2) 5x more than Good moves (class 0)
                    # to force the model to learn what a mistake looks like despite dataset imbalance.
                    q_weights = torch.tensor([1.0, 2.0, 5.0], device=device)
                    q_loss = F.cross_entropy(pred_q, batch_q, weight=q_weights) 
                    
                    # [Simple is Genius] Negative Reinforcement
                    # If opponent's prev move was a blunder (q=2), current player is winning.
                    # Penalize if it predicts a loss/draw (v < 0.30) in this highly favorable state.
                    blunder_mask = (batch_q == 2).to(torch.float32).view(-1, 1)
                    penalty = (blunder_mask * F.relu(0.30 - pred_v)).mean() * 2.0 # [NORMALIZED] Reduced from 10.0 to 2.0
                    
                    # Combined Loss: Value + Policy + Quality + Penalty - (Beta * Entropy)
                    loss = v_loss + p_loss + q_loss + penalty - (0.001 * entropy)
                
                # AMP Scaled Backward & Step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
                scaler.step(optimizer)
                scaler.update()
                
                total_loss_val += loss.item()
                total_p_loss += p_loss.item()
                total_v_loss += v_loss.item()
            
            # Step the combined scheduler (Warmup -> Cosine Decay)
            scheduler.step()
            
            # Performance Metrics (Averaged over the interval)
            train_duration = time.time() - train_start
            avg_loss = total_loss_val / dynamic_steps
            avg_v = total_v_loss / dynamic_steps
            avg_p = total_p_loss / dynamic_steps
            train_speed = (BATCH_SIZE * dynamic_steps) / max(0.001, train_duration)
            div_p = (diversity_samples.value / max(1, current_total)) * 100
            buffer_p = (current_count / MAX_SAMPLES) * 100
            
            # Detailed Per-Episode Reporting
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[EP] {episode} | Total: {current_total:,} | Speed: {train_speed:.1f}/s | Loss: {avg_loss:.4f} (V:{avg_v:.4f} P:{avg_p:.4f}) | GenSPS: {gen_speed:.1f}/s | Div: {div_p:.1f}% | Buffer: {buffer_p:.1f}% | LR: {current_lr:.6f}")

            if episode % 1000 == 0:
                # Save Model Only (Fast)
                save_state(f"{CHECKPOINT_DIR}/model_ep_{episode}.pth", full=False)
                
            # Periodic Full-State Checkpointing (Every 5000 Episodes)
            if episode % 5000 == 0:
                save_state(f"{CHECKPOINT_DIR}/full_state_ep_{episode}.pth", full=True)
                
                # No cleanup - keep all checkpoints as requested
                pass


    except KeyboardInterrupt:
        print("\nTraining interrupted. Shutting down workers...")
    finally:
        # Cleanup: Ensure all sub-processes and their children are terminated
        for p in workers: _kill_worker_tree(p.pid)
        print("\n[STOP] Training stopped. Performing FINAL backup...")
        save_state(f"{CHECKPOINT_DIR}/training_state.pth", full=True)
        print("[DONE] Shutdown complete.")

if __name__ == "__main__":
    train() # Launch training system

if __name__ == "__main__":
    train() # Launch training system
