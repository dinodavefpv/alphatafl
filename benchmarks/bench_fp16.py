"""
FP16 Mixed Precision Inference Benchmark.

Compares FP32 vs FP16 vs AMP autocast inference throughput and accuracy
for the AlphaTaflNet model across various batch sizes.

Usage:
    python benchmarks/bench_fp16.py [--batch-sizes 1,16,64,128,256] [--iterations 100]
"""
import os
import sys
import time
import argparse
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.model.network import AlphaTaflNet


def warmup(model, input_tensor, device, iterations=10):
    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(input_tensor)


def benchmark_inference(model, input_tensor, device, iterations=100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iterations * 1000


def main():
    parser = argparse.ArgumentParser(description="FP16 mixed precision inference benchmark")
    parser.add_argument('--batch-sizes', type=str, default='1,16,64,128,256',
                        help='Comma-separated batch sizes to test')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of forward passes per config')
    parser.add_argument('--model-path', type=str, default='models/current_best.pt')
    args = parser.parse_args()

    batch_sizes = [int(s) for s in args.batch_sizes.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("AlphaTafl FP16 vs FP32 Inference Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Iterations per config: {args.iterations}")
    print()

    fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    if not fp16_supported:
        print("WARNING: FP16 not supported on this device (requires compute capability 7.0+)")
        print("Will benchmark FP32 only.\n")

    results = []

    for batch_size in batch_sizes:
        print(f"--- Batch Size: {batch_size} ---")

        input_tensor = torch.randn(batch_size, 14, 11, 11, device=device)

        # === FP32 Baseline ===
        model_fp32 = AlphaTaflNet().to(device).eval()
        if os.path.exists(args.model_path):
            try:
                model_fp32.load_state_dict(torch.load(args.model_path, map_location=device))
            except Exception as e:
                print(f"  Could not load model: {e}, using random weights")

        # Capture baseline FP32 output for accuracy comparison
        with torch.inference_mode():
            baseline_policy, baseline_value = model_fp32(input_tensor)

        warmup(model_fp32, input_tensor, device, 10)
        fp32_time = benchmark_inference(model_fp32, input_tensor, device, args.iterations)
        fp32_throughput = (batch_size * args.iterations) / (fp32_time / 1000)
        fp32_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(f"  FP32:       {fp32_time:.3f} ms/batch  ({fp32_throughput:.0f} st/s)  {fp32_mem:.1f} MB VRAM")

        # === FP16 (half precision) ===
        fp16_time = None
        fp16_diff_policy = None
        fp16_diff_value = None
        fp16_mem = 0

        if fp16_supported:
            torch.cuda.reset_peak_memory_stats(device)
            model_fp16 = AlphaTaflNet().to(device).half().eval()
            if os.path.exists(args.model_path):
                try:
                    model_fp16.load_state_dict(torch.load(args.model_path, map_location=device))
                except Exception:
                    pass

            input_fp16 = input_tensor.half()

            warmup(model_fp16, input_fp16, device, 10)
            fp16_time = benchmark_inference(model_fp16, input_fp16, device, args.iterations)
            fp16_throughput = (batch_size * args.iterations) / (fp16_time / 1000)
            fp16_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            with torch.inference_mode():
                fp16_policy, fp16_value = model_fp16(input_fp16)
            fp16_diff_policy = (baseline_policy.float() - fp16_policy.float()).abs().max().item()
            fp16_diff_value = (baseline_value.float() - fp16_value.float()).abs().max().item()

            speedup = fp32_time / fp16_time
            print(f"  FP16:       {fp16_time:.3f} ms/batch  ({fp16_throughput:.0f} st/s)  {fp16_mem:.1f} MB VRAM  ({speedup:.2f}x speedup)")
            print(f"  FP16 acc:   policy max diff = {fp16_diff_policy:.6f}  value max diff = {fp16_diff_value:.6f}")

        # === AMP Mixed Precision (autocast) ===
        amp_time = None
        amp_diff_policy = None
        amp_diff_value = None
        amp_mem = 0

        if fp16_supported:
            torch.cuda.reset_peak_memory_stats(device)
            model_amp = AlphaTaflNet().to(device).eval()
            if os.path.exists(args.model_path):
                try:
                    model_amp.load_state_dict(torch.load(args.model_path, map_location=device))
                except Exception:
                    pass

            warmup(model_amp, input_tensor, device, 10)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    for _ in range(args.iterations):
                        _ = model_amp(input_tensor)
            torch.cuda.synchronize()
            amp_time = (time.perf_counter() - t0) / args.iterations * 1000
            amp_throughput = (batch_size * args.iterations) / (amp_time / 1000)
            amp_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    amp_policy, amp_value = model_amp(input_tensor)
            amp_diff_policy = (baseline_policy.float() - amp_policy.float()).abs().max().item()
            amp_diff_value = (baseline_value.float() - amp_value.float()).abs().max().item()

            speedup = fp32_time / amp_time
            print(f"  AMP:        {amp_time:.3f} ms/batch  ({amp_throughput:.0f} st/s)  {amp_mem:.1f} MB VRAM  ({speedup:.2f}x speedup)")
            print(f"  AMP acc:    policy max diff = {amp_diff_policy:.6f}  value max diff = {amp_diff_value:.6f}")

        results.append({
            'batch_size': batch_size,
            'fp32_ms': fp32_time,
            'fp32_st_s': fp32_throughput,
            'fp32_mem_mb': fp32_mem,
            'fp16_ms': fp16_time,
            'fp16_speedup': fp32_time / fp16_time if fp16_time else None,
            'fp16_max_diff_policy': fp16_diff_policy,
            'fp16_max_diff_value': fp16_diff_value,
            'fp16_mem_mb': fp16_mem,
            'amp_ms': amp_time,
            'amp_speedup': fp32_time / amp_time if amp_time else None,
            'amp_max_diff_policy': amp_diff_policy,
            'amp_max_diff_value': amp_diff_value,
            'amp_mem_mb': amp_mem,
        })

        del model_fp32
        if fp16_supported:
            del model_fp16
            del model_amp
        torch.cuda.empty_cache()
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Batch':>6} | {'FP32 ms':>8} {'st/s':>8} |"
    if fp16_supported:
        header += f" {'FP16 ms':>8} {'st/s':>8} {'spdup':>5} {'diff':>8} |"
        header += f" {'AMP ms':>8} {'st/s':>8} {'spdup':>5} {'diff':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r['batch_size']:>6} | {r['fp32_ms']:>8.3f} {r['fp32_st_s']:>8.0f} |"
        if fp16_supported:
            row += f" {r['fp16_ms']:>8.3f} {r['fp32_st_s'] * r['fp16_speedup']:>8.0f} {r['fp16_speedup']:>5.2f}x {r['fp16_max_diff_policy']:>8.6f} |"
            row += f" {r['amp_ms']:>8.3f} {r['fp32_st_s'] * r['amp_speedup']:>8.0f} {r['amp_speedup']:>5.2f}x {r['amp_max_diff_policy']:>8.6f}"
        print(row)

    torch.cuda.reset_peak_memory_stats(device)


if __name__ == '__main__':
    main()
