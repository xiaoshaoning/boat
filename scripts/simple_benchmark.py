#!/usr/bin/env python3
"""
Simple benchmark comparing mnist vs mnist_autodiff execution time.
"""

import os
import subprocess
import time
import sys
from pathlib import Path

def run_timing_test(executable_path, env=None, name="", cwd=None):
    """Run executable and return execution time in seconds."""
    if not executable_path.exists():
        print(f"Error: {executable_path} not found")
        return None

    print(f"Running {name} ({executable_path.name})...")

    # Ensure we use small dataset
    env = env or os.environ.copy()
    if 'USE_FULL_DATA' in env:
        del env['USE_FULL_DATA']

    # If cwd not provided, calculate from executable path
    if cwd is None:
        # executable_path is build/examples/mnist/Debug/mnist.exe
        # Need to go up 5 levels to get project root
        project_root = executable_path.parent.parent.parent.parent.parent
        cwd = project_root / "examples" / "mnist"

    start_time = time.time()
    try:
        result = subprocess.run(
            [str(executable_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(cwd)
        )
        elapsed = time.time() - start_time

        print(f"  Exit code: {result.returncode}")
        print(f"  Time: {elapsed:.2f}s")

        # Check for errors
        if result.returncode != 0:
            print(f"  Error: program exited with code {result.returncode}")
            if result.stderr:
                print(f"  Stderr (first 200 chars): {result.stderr[:200]}")

        # Check for NaN/Inf warnings in autodiff version
        if "autodiff" in name.lower() and result.stdout:
            if "NaN/Inf gradient values detected" in result.stdout or "nan_inf=" in result.stdout:
                # Extract nan_inf count
                import re
                nan_inf_matches = re.findall(r'nan_inf=(\d+)', result.stdout)
                if nan_inf_matches:
                    total = sum(int(x) for x in nan_inf_matches)
                    print(f"  NaN/Inf gradient values detected: {total}")

        return elapsed

    except subprocess.TimeoutExpired:
        print(f"  Timeout expired (5 minutes)")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    project_root = Path(__file__).parent
    build_dir = project_root / "build" / "examples" / "mnist" / "Debug"
    mnist_cwd = project_root / "examples" / "mnist"

    mnist_exe = build_dir / "mnist.exe"
    mnist_autodiff_exe = build_dir / "mnist_autodiff.exe"

    if not mnist_exe.exists():
        print(f"Error: {mnist_exe} not found")
        return 1

    if not mnist_autodiff_exe.exists():
        print(f"Error: {mnist_autodiff_exe} not found")
        return 1

    print("Simple Boat MNIST Benchmark")
    print("=" * 50)

    # Run each program once
    base_time = run_timing_test(mnist_exe, name="Base version (mnist)", cwd=mnist_cwd)
    print()

    autodiff_time = run_timing_test(mnist_autodiff_exe, name="Autodiff version (mnist_autodiff)", cwd=mnist_cwd)
    print()

    # Compare results
    print("=" * 50)
    print("Comparison Results")
    print("=" * 50)

    if base_time is not None and autodiff_time is not None:
        print(f"Base version time:     {base_time:.2f}s")
        print(f"Autodiff version time: {autodiff_time:.2f}s")

        if base_time > 0:
            ratio = autodiff_time / base_time
            difference = autodiff_time - base_time
            print(f"\nAutodiff is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than Base")
            print(f"Time difference: {difference:+.2f}s")

            if ratio > 1:
                overhead_percent = (ratio - 1) * 100
                print(f"Autodiff overhead: {overhead_percent:.1f}%")
        else:
            print("Base time is zero, cannot compute ratio")

        # Save results
        results_file = project_root / "simple_benchmark_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"Base version time: {base_time:.2f}s\n")
            f.write(f"Autodiff version time: {autodiff_time:.2f}s\n")
            if base_time > 0:
                f.write(f"Ratio (autodiff/base): {autodiff_time/base_time:.2f}\n")
                f.write(f"Time difference: {autodiff_time-base_time:+.2f}s\n")

        print(f"\nResults saved to: {results_file}")
    else:
        print("Benchmark incomplete due to errors.")

    return 0

if __name__ == "__main__":
    sys.exit(main())