"""
Multi-worker inference benchmark.

Measures self-play throughput (states/sec, games/hour) with concurrent workers
sharing a single GPU inference server. Tests different worker counts and
inference timeout settings to find the optimal configuration.

Usage:
    python benchmarks/bench_multi_worker.py [--workers 1,2,4,8,12,20] [--timeouts 0,5] [--duration 60]
"""
import os
import sys
import time
import json
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def run_config(num_workers, timeout_ms, simulations=50, batch_size=16, duration=60, use_shm=False, max_batch=256):
    """Run a single benchmark configuration. Returns dict or None on failure."""
    helper = os.path.join(PROJECT_ROOT, "benchmarks", "_bench_multi_worker_helper.py")

    env = os.environ.copy()
    env['BENCH_NUM_WORKERS'] = str(num_workers)
    env['BENCH_NUM_SIMULATIONS'] = str(simulations)
    env['BENCH_BATCH_SIZE'] = str(batch_size)
    env['BENCH_DURATION'] = str(duration)
    env['ALPHATAFL_INFERENCE_TIMEOUT_MS'] = str(timeout_ms)
    env['ALPHATAFL_INFERENCE_MAX_BATCH'] = str(max_batch)
    env['ALPHATAFL_BATCHED'] = '1'
    env['ALPHATAFL_SHM'] = '1' if use_shm else '0'

    timeout = duration + 120
    try:
        result = subprocess.run(
            [sys.executable, helper],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
            timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None

    for line in result.stdout.splitlines():
        if line.startswith("RESULT "):
            parts = line.split()
            games = int(parts[1])
            states = int(parts[2])
            actual_duration = float(parts[3])
            shm_flag = parts[6] if len(parts) >= 7 else "0"
            return {
                'num_workers': num_workers,
                'timeout_ms': timeout_ms,
                'use_shm': shm_flag == "1",
                'sims': simulations,
                'batch_size': batch_size,
                'games': games,
                'states': states,
                'duration': actual_duration,
                'states_per_sec': states / actual_duration if actual_duration > 0 else 0,
                'games_per_hour': (games / actual_duration) * 3600 if actual_duration > 0 else 0,
            }

    print(f"  FAILED. stderr tail: {result.stderr[-300:] if result.stderr else 'none'}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Multi-worker self-play throughput benchmark")
    parser.add_argument('--workers', type=str, default='1,2,4,8,12,20',
                        help='Comma-separated worker counts to test')
    parser.add_argument('--timeouts', type=str, default='0,5',
                        help='Comma-separated timeout values in ms')
    parser.add_argument('--duration', type=int, default=60,
                        help='Benchmark duration per config (seconds)')
    parser.add_argument('--simulations', type=int, default=50,
                        help='MCTS simulations per turn (single value for classic mode)')
    parser.add_argument('--sims', type=str, default=None,
                        help='Comma-separated sim counts for batch-sweep mode (e.g. "50,256,800")')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='MCTS batch size (single value for classic mode)')
    parser.add_argument('--batch-sizes', type=str, default=None,
                        help='Comma-separated batch sizes for batch-sweep mode (e.g. "8,16,32,64,128")')
    parser.add_argument('--max-batch', type=int, default=256,
                        help='Inference server max batch (ALPHATAFL_INFERENCE_MAX_BATCH)')
    parser.add_argument('--shm', action='store_true',
                        help='Compare shared memory transport vs mp.Queue')
    args = parser.parse_args()

    worker_counts = [int(w) for w in args.workers.split(',')]
    timeouts = [int(t) for t in args.timeouts.split(',')]

    # Batch-sweep mode: --sims and --batch-sizes provide multi-value iteration
    batch_sweep_mode = args.sims is not None and args.batch_sizes is not None
    if batch_sweep_mode:
        sim_counts = [int(s) for s in args.sims.split(',')]
        batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
        # In sweep mode, use fixed workers/timeout, SHM only
        assert len(worker_counts) == 1, "--workers must be single value in batch-sweep mode"
        assert len(timeouts) == 1, "--timeouts must be single value in batch-sweep mode"
        num_workers = worker_counts[0]
        timeout_ms = timeouts[0]

    import torch
    has_cuda = torch.cuda.is_available()
    print("=" * 75)
    print("AlphaTafl Multi-Worker Throughput Benchmark")
    print("=" * 75)
    print(f"GPU: {'CUDA available' if has_cuda else 'CPU-only'}")
    if has_cuda:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    if batch_sweep_mode:
        print(f"Mode: Batch-size sweep (SHM, {num_workers} workers, {timeout_ms}ms timeout, max_batch={args.max_batch})")
        print(f"Sim counts: {sim_counts}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Window: {args.duration}s per config")
    else:
        print(f"Config: {args.simulations} sims/turn, batch_size={args.batch_size}, {args.duration}s/window")
        print(f"Workers: {worker_counts}")
        print(f"Timeouts: {timeouts}ms")
        print(f"Transport: {'SHM vs Queue comparison' if args.shm else 'mp.Queue only'}")

    results = []

    if batch_sweep_mode:
        for sims in sim_counts:
            print(f"\n{'-'*50}")
            print(f"  {sims} sims/turn")
            print(f"{'-'*50}")
            for bs in batch_sizes:
                label = f"  bs={bs:>4}"
                print(f"[{label}] Running...", end=' ', flush=True)
                r = run_config(num_workers, timeout_ms, simulations=sims, batch_size=bs,
                               duration=args.duration, use_shm=True, max_batch=args.max_batch)
                if r:
                    results.append(r)
                    print(f"{r['states']:>5} st, {r['states_per_sec']:>6.0f} st/s")
                else:
                    print("FAILED")

    else:
        # Classic mode
        for nw in worker_counts:
            for to in timeouts:
                label = f"{nw}w {to}ms queue"
                print(f"\n[{label}] Running...", end=' ', flush=True)
                r = run_config(nw, to, args.simulations, args.batch_size, args.duration, use_shm=False, max_batch=args.max_batch)
                if r:
                    results.append(r)
                    print(f"{r['games']} games, {r['states_per_sec']:.0f} st/s")
                else:
                    print("FAILED")

                if args.shm:
                    label = f"{nw}w {to}ms SHM"
                    print(f"[{label}] Running...", end=' ', flush=True)
                    r = run_config(nw, to, args.simulations, args.batch_size, args.duration, use_shm=True, max_batch=args.max_batch)
                    if r:
                        results.append(r)
                        print(f"{r['games']} games, {r['states_per_sec']:.0f} st/s")
                    else:
                        print("FAILED")

    if not results:
        print("\nNo results collected.")
        return

    if batch_sweep_mode:
        # Batch-sweep output: table per sim count, rows=batch_sizes
        print(f"\n{'='*75}")
        print(f"BATCH SIZE SWEEP — {num_workers} workers, SHM, {timeout_ms}ms timeout")
        print(f"{'='*75}")
        print(f"{'Batch':>6} | {'Batch/s/turn':>12} |", end='')
        for sims in sim_counts:
            print(f" {sims:>4} sims |", end='')
        print()
        print("-" * (21 + 14 * len(sim_counts)))

        for bs in batch_sizes:
            row = f"{bs:>6} |"
            # Calculate round-trips per turn
            sims_for_rt = sim_counts[0] if sim_counts else 50
            rts = max(1, (sims_for_rt + bs - 1) // bs)
            row += f" {rts:>11} |"
            for sims in sim_counts:
                r = next((r for r in results if r.get('sims') == sims and r['num_workers'] == num_workers), None)
                # Actually need to match batch_size too — but we only stored num_workers. 
                # Need to also store batch_size in result. Let me do inline lookup differently.
                m = [r2 for r2 in results if r2.get('sims') == sims]
                r = next((r2 for r2 in m if r2.get('batch_size') == bs), None)
                if r:
                    row += f" {r['states_per_sec']:>6.0f} |"
                else:
                    row += f" {'N/A':>6} |"
            print(row)

        # Projected 800-sim game time
        print(f"\n{'='*75}")
        print(f"PROJECTED 180-TURN GAME TIME AT 800 SIMS/TURN")
        print(f"{'='*75}")
        header = f"{'Batch':>6} | Rnd-trips(800) |"
        for sims in sim_counts:
            header += f" via@{sims:>4} |"
        print(header)
        print("-" * len(header))
        for bs in batch_sizes:
            row = f"{bs:>6} |"
            rts_800 = max(1, (800 + bs - 1) // bs)
            row += f" {rts_800:>13} |"
            for sims in sim_counts:
                r = next((r2 for r2 in results if r2.get('sims') == sims and r2.get('batch_size') == bs and r2['states_per_sec'] > 0), None)
                if r:
                    batch_rate = r['states_per_sec'] * max(1, (sims + bs - 1) // bs)
                    game_time = 180 * rts_800 / batch_rate
                    row += f" {game_time:>7.1f}s |"
                else:
                    row += f" {'N/A':>7} |"
            print(row)

        # Redundancy check: highlight where batch_size exceeds sims
        print(f"\n{'-'*50}")
        print("Redundancy markers: (* = batch_size >= sims, all identical = 1 batch/turn)")
        for sims in sim_counts:
            same = []
            for bs in batch_sizes:
                if bs >= sims:
                    same.append(str(bs))
            if same:
                print(f"  {sims} sims: batch sizes {', '.join(same)} should produce identical throughput")

    else:
        # Classic output (unchanged)
        transport_modes = ['queue']
        if args.shm:
            transport_modes = ['queue', 'SHM']
        for transport in transport_modes:
            t_results = [r for r in results if r['use_shm'] == (transport == 'SHM')]
            if not t_results:
                continue
            print(f"\n{'='*75}")
            print(f"THROUGHPUT: states/sec — {transport.upper()} transport")
            print(f"{'='*75}")
            header = f"{'Workers':>8} |"
            for to in timeouts:
                header += f" {to:>4}ms |"
            print(header)
            print("-" * len(header))
            for nw in worker_counts:
                row = f"{nw:>8} |"
                for to in timeouts:
                    r = next((r for r in t_results if r['num_workers'] == nw and r['timeout_ms'] == to), None)
                    if r:
                        row += f" {r['states_per_sec']:>6.0f} |"
                    else:
                        row += f" {'N/A':>6} |"
                print(row)

        if args.shm:
            print(f"\n{'='*75}")
            print("SHM SPEEDUP: SHM throughput / Queue throughput")
            print(f"{'='*75}")
            header = f"{'Workers':>8} |"
            for to in timeouts:
                header += f" {to:>4}ms |"
            print(header)
            print("-" * len(header))
            for nw in worker_counts:
                row = f"{nw:>8} |"
                for to in timeouts:
                    r_q = next((r for r in results if r['num_workers'] == nw and r['timeout_ms'] == to and not r['use_shm']), None)
                    r_s = next((r for r in results if r['num_workers'] == nw and r['timeout_ms'] == to and r['use_shm']), None)
                    if r_q and r_s and r_q['states_per_sec'] > 0:
                        speedup = r_s['states_per_sec'] / r_q['states_per_sec']
                        row += f" {speedup:>5.1f}x |"
                    else:
                        row += f" {'N/A':>6} |"
                print(row)


if __name__ == '__main__':
    main()
