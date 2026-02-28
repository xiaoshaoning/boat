#!/usr/bin/env python3
"""
Convergence test for mnist_autodiff on full dataset.
"""

import os
import subprocess
import time
import sys
from pathlib import Path

def run_convergence_test():
    project_root = Path(__file__).parent
    executable = project_root / "build" / "examples" / "mnist" / "Debug" / "mnist_autodiff.exe"

    if not executable.exists():
        print(f"Error: {executable} not found")
        return False

    print("Running convergence test on full MNIST dataset...")
    print(f"Executable: {executable}")

    # Set environment variable to use full dataset
    env = os.environ.copy()
    env['USE_FULL_DATA'] = '1'

    start_time = time.time()
    try:
        result = subprocess.run(
            [str(executable)],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=project_root / "examples" / "mnist"
        )
        elapsed = time.time() - start_time

        print(f"Exit code: {result.returncode}")
        print(f"Elapsed time: {elapsed:.2f}s")

        # Parse output to check convergence
        lines = result.stdout.split('\n')

        # Look for epoch loss values
        losses = []
        accuracies = []

        for line in lines:
            # Epoch format: Epoch 1/5: time=12.34s, loss=1.2345, accuracy=45.67%...
            if 'Epoch' in line and 'loss=' in line:
                import re
                match = re.search(r'loss=([\d.]+)', line)
                if match:
                    losses.append(float(match.group(1)))

                match = re.search(r'accuracy=([\d.]+)%', line)
                if match:
                    accuracies.append(float(match.group(1)))

        print(f"\nConvergence analysis:")
        print(f"  Number of epochs completed: {len(losses)}")

        if len(losses) >= 2:
            print(f"  Loss values: {losses}")
            print(f"  Accuracy values: {accuracies}")

            # Check if loss is decreasing
            loss_decreased = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
            if loss_decreased:
                print("  ✓ Loss is decreasing (monotonically)")
            else:
                print("  ⚠ Loss not monotonically decreasing")

            # Check final loss
            if losses[-1] < losses[0]:
                print(f"  ✓ Final loss ({losses[-1]:.4f}) is lower than initial loss ({losses[0]:.4f})")
                improvement = (losses[0] - losses[-1]) / losses[0] * 100
                print(f"    Improvement: {improvement:.1f}%")
            else:
                print(f"  ⚠ Final loss ({losses[-1]:.4f}) is not lower than initial loss ({losses[0]:.4f})")

            # Check accuracy trend
            if len(accuracies) >= 2:
                accuracy_increased = all(accuracies[i] <= accuracies[i+1] for i in range(len(accuracies)-1))
                if accuracy_increased:
                    print("  ✓ Accuracy is increasing (monotonically)")
                else:
                    print("  ⚠ Accuracy not monotonically increasing")

                if accuracies[-1] > accuracies[0]:
                    improvement = (accuracies[-1] - accuracies[0]) / accuracies[0] * 100
                    print(f"  ✓ Final accuracy ({accuracies[-1]:.2f}%) is higher than initial ({accuracies[0]:.2f}%)")
                    print(f"    Improvement: {improvement:.1f}%")

        # Look for validation results
        val_losses = []
        val_accuracies = []
        for line in lines:
            if 'validation loss=' in line:
                import re
                match = re.search(r'validation loss=([\d.]+)', line)
                if match:
                    val_losses.append(float(match.group(1)))
                match = re.search(r'accuracy=([\d.]+)%', line)
                if match:
                    val_accuracies.append(float(match.group(1)))

        if val_losses:
            print(f"\nValidation results:")
            print(f"  Validation losses: {val_losses}")
            print(f"  Validation accuracies: {val_accuracies}")

            if len(val_losses) >= 2 and val_losses[-1] < val_losses[0]:
                print(f"  ✓ Validation loss improved from {val_losses[0]:.4f} to {val_losses[-1]:.4f}")

        # Look for gradient statistics including NaN/Inf detection
        nan_inf_counts = []
        for line in lines:
            if 'nan_inf=' in line:
                import re
                match = re.search(r'nan_inf=(\d+)', line)
                if match:
                    nan_inf_counts.append(int(match.group(1)))

        if nan_inf_counts:
            total_nan_inf = sum(nan_inf_counts)
            if total_nan_inf > 0:
                print(f"\n⚠ WARNING: Total NaN/Inf gradient values detected: {total_nan_inf}")
            else:
                print(f"\n✓ No NaN/Inf gradient values detected")

        # Save output to file
        output_file = project_root / "convergence_test_output.txt"
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        print(f"\nFull output saved to: {output_file}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("Timeout expired (10 minutes)")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = run_convergence_test()
    sys.exit(0 if success else 1)