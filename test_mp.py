import multiprocessing as mp
import sys
import os

# Ensure the root directory is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.orchestrator import worker_task, init_worker

def run_test():
    print("Starting pool...")
    with mp.Pool(processes=1, initializer=init_worker) as pool:
        print("Dispatching task...")
        try:
            result = pool.apply(worker_task, args=((1, 5, "models/current_best.pt"),))
            print(f"Success! Result: {result}")
        except Exception as e:
            print(f"Caught exception: {e}")

if __name__ == '__main__':
    run_test()
