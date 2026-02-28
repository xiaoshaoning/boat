#!/usr/bin/env python3
"""
Benchmark script to compare mnist (base) and mnist_autodiff (autodiff) performance.
"""

import os
import subprocess
import time
import re
import sys
from pathlib import Path

def run_benchmark(executable_path, env=None, args=None):
    """
    Run benchmark and capture output.
    Returns dict with metrics.
    """
    cmd = [str(executable_path)]
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    if env:
        print(f"Environment: {env}")

    start_time = time.time()
    try:
        # Calculate correct working directory: examples/mnist
        # executable_path is build/examples/mnist/Debug/mnist.exe
        # Need to go up 5 levels to get project root
        project_root = Path(executable_path).parent.parent.parent.parent.parent
        mnist_cwd = project_root / "examples" / "mnist"

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(mnist_cwd)
        )
        elapsed = time.time() - start_time

        metrics = {
            'exit_code': result.returncode,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }

        # Filter out DEBUG lines before parsing
        filtered_stdout = '\n'.join([line for line in result.stdout.split('\n') if 'DEBUG' not in line])

        # Parse metrics from output
        parse_metrics_from_output(metrics, filtered_stdout)

        return metrics

    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {executable_path}")
        return {'timeout': True, 'success': False}
    except Exception as e:
        print(f"Error running {executable_path}: {e}")
        return {'error': str(e), 'success': False}

def parse_metrics_from_output(metrics, stdout):
    """
    Parse performance metrics from program output.
    """
    # Example output lines to parse:
    # Epoch 1/2: time=12.34s, loss=1.2345, accuracy=45.67%, throughput=1234 samples/s...
    # memory: allocated=12.34 MB, blocks=123, peak=56.78 MB
    # gradients: norm=1.234, max=5.678, clip_ratio=12.34%

    metrics['epoch_times'] = []
    metrics['losses'] = []
    metrics['accuracies'] = []
    metrics['throughputs'] = []
    metrics['memory_allocated'] = []
    metrics['memory_peak'] = []
    metrics['gradient_norms'] = []

    lines = stdout.split('\n')

    # Patterns for different output formats
    # Autodiff format: Epoch 1/5: time=12.34s, loss=1.2345, accuracy=45.67%...
    epoch_pattern_autodiff = re.compile(r'Epoch\s+(\d+)/(\d+):\s+time=([\d.]+)s,\s+loss=([\d.]+),\s+accuracy=([\d.]+)%')
    # Base format: Epoch 1/5: time=10.29s, accuracy=21.77%
    epoch_pattern_base = re.compile(r'Epoch\s+(\d+)/(\d+):\s+time=([\d.]+)s,\s+accuracy=([\d.]+)%')
    memory_pattern = re.compile(r'memory:\s+allocated=([\d.]+)\s+MB,\s+blocks=\d+,\s+peak=([\d.]+)\s+MB')
    gradient_pattern = re.compile(r'gradients:\s+norm=([\d.]+),\s+max=([\d.]+),\s+clip_ratio=([\d.]+)%')

    for line in lines:
        # Parse epoch information - try autodiff format first
        epoch_match = epoch_pattern_autodiff.search(line)
        if epoch_match:
            epoch_num, total_epochs, epoch_time, loss, accuracy = epoch_match.groups()
            metrics['epoch_times'].append(float(epoch_time))
            metrics['losses'].append(float(loss))
            metrics['accuracies'].append(float(accuracy))
        else:
            # Try base format
            epoch_match = epoch_pattern_base.search(line)
            if epoch_match:
                epoch_num, total_epochs, epoch_time, accuracy = epoch_match.groups()
                metrics['epoch_times'].append(float(epoch_time))
                metrics['losses'].append(None)  # No loss in base version
                metrics['accuracies'].append(float(accuracy))

        # If we found an epoch line (either format), look for throughput
        if epoch_match:
            throughput_match = re.search(r'throughput=([\d.]+)\s+samples/s', line)
            if throughput_match:
                metrics['throughputs'].append(float(throughput_match.group(1)))

        # Parse memory information
        memory_match = memory_pattern.search(line)
        if memory_match:
            allocated, peak = memory_match.groups()
            metrics['memory_allocated'].append(float(allocated))
            metrics['memory_peak'].append(float(peak))

        # Parse gradient information
        gradient_match = gradient_pattern.search(line)
        if gradient_match:
            norm, max_grad, clip_ratio = gradient_match.groups()
            metrics['gradient_norms'].append(float(norm))

    # Calculate summary statistics
    if metrics['epoch_times']:
        metrics['total_time'] = sum(metrics['epoch_times'])
        metrics['avg_epoch_time'] = sum(metrics['epoch_times']) / len(metrics['epoch_times'])

    if metrics['losses']:
        # Filter out None values (base version doesn't have loss)
        valid_losses = [loss for loss in metrics['losses'] if loss is not None]
        if valid_losses:
            metrics['final_loss'] = valid_losses[-1]
            metrics['best_loss'] = min(valid_losses)
        else:
            metrics['final_loss'] = None
            metrics['best_loss'] = None

    if metrics['accuracies']:
        metrics['final_accuracy'] = metrics['accuracies'][-1] if metrics['accuracies'] else None
        metrics['best_accuracy'] = max(metrics['accuracies']) if metrics['accuracies'] else None

    return metrics

def compare_results(base_results, autodiff_results):
    """
    Compare results between base and autodiff versions.
    """
    comparison = {}

    # Time comparison
    if 'total_time' in base_results and 'total_time' in autodiff_results:
        base_time = base_results['total_time']
        autodiff_time = autodiff_results['total_time']
        comparison['time_ratio'] = autodiff_time / base_time if base_time > 0 else float('inf')
        comparison['time_difference'] = autodiff_time - base_time

    # Memory comparison
    if 'memory_peak' in base_results and base_results['memory_peak'] and \
       'memory_peak' in autodiff_results and autodiff_results['memory_peak']:
        base_mem = max(base_results['memory_peak'])
        autodiff_mem = max(autodiff_results['memory_peak'])
        comparison['memory_ratio'] = autodiff_mem / base_mem if base_mem > 0 else float('inf')
        comparison['memory_difference'] = autodiff_mem - base_mem

    # Accuracy comparison
    if 'final_accuracy' in base_results and 'final_accuracy' in autodiff_results:
        base_acc = base_results['final_accuracy']
        autodiff_acc = autodiff_results['final_accuracy']
        comparison['accuracy_difference'] = autodiff_acc - base_acc

    # Loss comparison
    if 'final_loss' in base_results and 'final_loss' in autodiff_results:
        base_loss = base_results['final_loss']
        autodiff_loss = autodiff_results['final_loss']
        if base_loss is not None and autodiff_loss is not None:
            comparison['loss_difference'] = autodiff_loss - base_loss

    return comparison

def print_results(results, name):
    """
    Print benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {name}")
    print(f"{'='*60}")

    if not results.get('success', False):
        print(f"  FAILED: {results.get('error', 'Unknown error')}")
        return

    print(f"  Exit code: {results.get('exit_code', 'N/A')}")
    print(f"  Total elapsed time: {results.get('elapsed_time', 'N/A'):.2f}s")

    if 'total_time' in results:
        print(f"  Total training time: {results['total_time']:.2f}s")

    if 'avg_epoch_time' in results:
        print(f"  Average epoch time: {results['avg_epoch_time']:.2f}s")

    if 'final_accuracy' in results:
        print(f"  Final accuracy: {results['final_accuracy']:.2f}%")

    if 'final_loss' in results and results['final_loss'] is not None:
        print(f"  Final loss: {results['final_loss']:.4f}")

    if 'memory_peak' in results and results['memory_peak']:
        print(f"  Peak memory: {max(results['memory_peak']):.2f} MB")

    if 'gradient_norms' in results and results['gradient_norms']:
        print(f"  Final gradient norm: {results['gradient_norms'][-1] if results['gradient_norms'] else 'N/A':.4f}")

    # Print epoch details
    if results.get('epoch_times'):
        print(f"\n  Epoch details:")
        for i, (epoch_time, loss, acc) in enumerate(zip(
            results['epoch_times'],
            results.get('losses', [None]*len(results['epoch_times'])),
            results.get('accuracies', [None]*len(results['epoch_times']))
        )):
            print(f"    Epoch {i+1}: time={epoch_time:.2f}s, "
                  f"loss={loss if loss is not None else 'N/A'}, "
                  f"accuracy={acc if acc is not None else 'N/A'}%")

def main():
    # Paths to executables
    project_root = Path(__file__).parent
    build_dir = project_root / "build" / "examples" / "mnist" / "Debug"

    mnist_exe = build_dir / "mnist.exe"
    mnist_autodiff_exe = build_dir / "mnist_autodiff.exe"

    if not mnist_exe.exists():
        print(f"Error: {mnist_exe} not found")
        return 1

    if not mnist_autodiff_exe.exists():
        print(f"Error: {mnist_autodiff_exe} not found")
        return 1

    print("Boat MNIST Benchmark Comparison")
    print(f"Base version: {mnist_exe}")
    print(f"Autodiff version: {mnist_autodiff_exe}")
    print()

    # Run base version (mnist)
    # Use small dataset (don't set USE_FULL_DATA environment variable)
    base_env = os.environ.copy()
    # Ensure we don't use full dataset
    if 'USE_FULL_DATA' in base_env:
        del base_env['USE_FULL_DATA']

    print("Running base version (mnist)...")
    base_results = run_benchmark(mnist_exe, env=base_env)
    print_results(base_results, "Base Version (mnist)")

    # Run autodiff version (mnist_autodiff)
    autodiff_env = os.environ.copy()
    # Use small dataset for autodiff too
    if 'USE_FULL_DATA' in autodiff_env:
        del autodiff_env['USE_FULL_DATA']

    print("\nRunning autodiff version (mnist_autodiff)...")
    autodiff_results = run_benchmark(mnist_autodiff_exe, env=autodiff_env)
    print_results(autodiff_results, "Autodiff Version (mnist_autodiff)")

    # Compare results
    if base_results.get('success') and autodiff_results.get('success'):
        comparison = compare_results(base_results, autodiff_results)

        print(f"\n{'='*60}")
        print("Performance Comparison")
        print(f"{'='*60}")

        if 'time_ratio' in comparison:
            ratio = comparison['time_ratio']
            diff = comparison['time_difference']
            print(f"Training time: Autodiff is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than Base")
            print(f"               (Difference: {diff:+.2f}s)")

        if 'memory_ratio' in comparison:
            ratio = comparison['memory_ratio']
            diff = comparison['memory_difference']
            print(f"Peak memory:   Autodiff uses {ratio:.2f}x {'more' if ratio > 1 else 'less'} memory than Base")
            print(f"               (Difference: {diff:+.2f} MB)")

        if 'accuracy_difference' in comparison:
            diff = comparison['accuracy_difference']
            print(f"Final accuracy: Autodiff is {diff:+.2f}% {'higher' if diff > 0 else 'lower'} than Base")

        if 'loss_difference' in comparison:
            diff = comparison['loss_difference']
            print(f"Final loss:     Autodiff is {diff:+.4f} {'higher' if diff > 0 else 'lower'} than Base")

    # Save results to file
    output_file = project_root / "mnist_benchmark_results.txt"
    with open(output_file, 'w') as f:
        f.write("Boat MNIST Benchmark Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Base Version (mnist):\n")
        if base_results.get('success'):
            f.write(f"  Success: Yes\n")
            f.write(f"  Total time: {base_results.get('total_time', 'N/A')}s\n")
            f.write(f"  Final accuracy: {base_results.get('final_accuracy', 'N/A')}%\n")
            f.write(f"  Final loss: {base_results.get('final_loss', 'N/A')}\n")
        else:
            f.write(f"  Success: No\n")
            f.write(f"  Error: {base_results.get('error', 'Unknown')}\n")

        f.write("\nAutodiff Version (mnist_autodiff):\n")
        if autodiff_results.get('success'):
            f.write(f"  Success: Yes\n")
            f.write(f"  Total time: {autodiff_results.get('total_time', 'N/A')}s\n")
            f.write(f"  Final accuracy: {autodiff_results.get('final_accuracy', 'N/A')}%\n")
            f.write(f"  Final loss: {autodiff_results.get('final_loss', 'N/A')}\n")
        else:
            f.write(f"  Success: No\n")
            f.write(f"  Error: {autodiff_results.get('error', 'Unknown')}\n")

        if base_results.get('success') and autodiff_results.get('success'):
            f.write("\nComparison:\n")
            for key, value in comparison.items():
                f.write(f"  {key}: {value}\n")

    print(f"\nResults saved to: {output_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())