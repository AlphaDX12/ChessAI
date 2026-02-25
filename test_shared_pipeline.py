import multiprocessing
import numpy as np
import time
import torch
import chess.engine
import os
import chess_engine

def test_shared_memory_pipeline():
    print("[TEST] Testing High-Performance Shared Memory Pipeline...")
    
    from chess_engine import OBS_SHAPE, MOVE_ACTION_DIM
    
    # 1. Setup Shared Memory (matches train_az.py logic)
    size = 1000
    
    s_arr = multiprocessing.RawArray('H', size * OBS_SHAPE[0] * 8 * 8)
    p_arr = multiprocessing.RawArray('H', size * MOVE_ACTION_DIM) 
    v_arr = multiprocessing.RawArray('H', size)
    q_arr = multiprocessing.RawArray('B', size)
    
    ptr = multiprocessing.Value('i', 0)
    count = multiprocessing.Value('i', 0)
    total_samples = multiprocessing.Value('L', 0)
    diversity_samples = multiprocessing.Value('L', 0)
    lock = multiprocessing.Lock()
    panic_flag = multiprocessing.Value('d', 0.0)
    
    # 2. Spawn one worker
    p = multiprocessing.Process(
        target=chess_engine.run_game_worker,
        args=(0, panic_flag, s_arr, p_arr, v_arr, q_arr, ptr, count, total_samples, diversity_samples, lock)
    )
    p.daemon = True
    p.start()
    
    print("[WAIT] Waiting for worker to generate some data...")
    start_time = time.time()
    generated = False
    while time.time() - start_time < 60:
        if count.value > 0:
            print(f"[DONE] Data detected! Count: {count.value} | Ptr: {ptr.value} | Total: {total_samples.value}")
            generated = True
            break
        time.sleep(1)
        
    if not generated:
        print("[ERROR] Failed: No data generated in 60 seconds.")
        p.terminate()
        return
 
    # 3. Verify Data format
    states = np.frombuffer(s_arr, dtype=np.float16).reshape(size, *OBS_SHAPE)
    policies = np.frombuffer(p_arr, dtype=np.float16).reshape(size, MOVE_ACTION_DIM)
    values = np.frombuffer(v_arr, dtype=np.float16).reshape(size, 1)
    qualities = np.frombuffer(q_arr, dtype=np.uint8).reshape(size, 1)
    
    # Check first sample
    idx = 0
    s0 = states[idx]
    p0 = policies[idx]
    v0 = values[idx]
    q0 = qualities[idx]
    
    print(f"[INFO] Sample Check:")
    print(f"   State Shape: {s0.shape} (Expected {OBS_SHAPE})")
    print(f"   State Dtype: {s0.dtype}")
    print(f"   Policy Sum:  {p0.sum():.2f}")
    print(f"   Value:       {v0[0]}")
    
    if s0.sum() != 0:
        print("[RUN] Pipeline verification SUCCESS!")
    else:
        print("[ERROR] Pipeline verification FAILED: Invalid data content.")
        
    p.terminate()

if __name__ == "__main__":
    test_shared_memory_pipeline()
