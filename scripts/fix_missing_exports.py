#!/usr/bin/env python3
"""
Script to fix missing BOAT_API exports based on actual linker errors.
Uses the list of missing symbols extracted from linker errors.
"""

import os
import re
import sys

# List of missing symbols from linker errors (collected from build output)
MISSING_SYMBOLS = [
    "boat_tensor_create",
    "boat_tensor_create_like",
    "boat_tensor_from_data",
    "boat_tensor_unref",
    "boat_tensor_shape",
    "boat_tensor_ndim",
    "boat_tensor_dtype",
    "boat_tensor_nbytes",
    "boat_tensor_nelements",
    "boat_tensor_data",
    "boat_tensor_const_data",
    "boat_tensor_slice",
    "boat_dtype_size",
    "boat_variable_create",
    "boat_variable_free",
    "boat_variable_data",
    "boat_variable_reset_data",
    "boat_variable_backward_full",
    "boat_var_mul",
    "boat_var_conv",
    "boat_var_pool",
    "boat_var_flatten",
    "boat_var_dense",
    "boat_var_relu",
    "boat_var_log_softmax",
    "boat_var_sum",
    "boat_relu",
    "boat_dense_layer_create",
    "boat_dense_layer_free",
    "boat_dense_layer_forward",
    "boat_dense_layer_get_bias",
    "boat_dense_layer_get_grad_bias",
    "boat_dense_layer_get_grad_weight",
    "boat_dense_layer_get_weight",
    "boat_dense_layer_set_bias",
    "boat_dense_layer_set_weight",
    "boat_conv_layer_create",
    "boat_conv_layer_forward",
    "boat_conv_layer_free",
    "boat_conv_layer_get_bias",
    "boat_conv_layer_get_grad_bias",
    "boat_conv_layer_get_grad_weight",
    "boat_conv_layer_get_weight",
    "boat_conv_layer_set_bias",
    "boat_conv_layer_set_weight",
    "boat_pool_layer_create",
    "boat_pool_layer_forward",
    "boat_pool_layer_free",
    "boat_flatten_layer_create",
    "boat_flatten_layer_forward",
    "boat_flatten_layer_free",
    "boat_adam_optimizer_create",
    "boat_optimizer_add_parameter",
    "boat_optimizer_free",
    "boat_optimizer_get_learning_rate",
    "boat_optimizer_set_learning_rate",
    "boat_optimizer_step",
    "boat_optimizer_zero_grad",
    "boat_cosine_annealing_scheduler_create",
    "boat_scheduler_free",
    "boat_scheduler_step",
    "boat_scheduler_update_optimizer",
    "boat_memory_get_stats",
    "boat_memory_reset_stats",
]

# Map symbols to likely source files
SYMBOL_TO_FILE = {
    # tensor.c symbols
    "boat_tensor_create": "src/core/tensor.c",
    "boat_tensor_create_like": "src/core/tensor.c",
    "boat_tensor_from_data": "src/core/tensor.c",
    "boat_tensor_unref": "src/core/tensor.c",
    "boat_tensor_shape": "src/core/tensor.c",
    "boat_tensor_ndim": "src/core/tensor.c",
    "boat_tensor_dtype": "src/core/tensor.c",
    "boat_tensor_nbytes": "src/core/tensor.c",
    "boat_tensor_nelements": "src/core/tensor.c",
    "boat_tensor_data": "src/core/tensor.c",
    "boat_tensor_const_data": "src/core/tensor.c",
    "boat_tensor_slice": "src/core/tensor.c",
    "boat_dtype_size": "src/core/tensor.c",

    # autodiff.c symbols
    "boat_variable_create": "src/autodiff.c",
    "boat_variable_free": "src/autodiff.c",
    "boat_variable_data": "src/autodiff.c",
    "boat_variable_reset_data": "src/autodiff.c",
    "boat_variable_backward_full": "src/autodiff.c",
    "boat_var_mul": "src/autodiff.c",
    "boat_var_conv": "src/autodiff.c",
    "boat_var_pool": "src/autodiff.c",
    "boat_var_flatten": "src/autodiff.c",
    "boat_var_dense": "src/autodiff.c",
    "boat_var_relu": "src/autodiff.c",
    "boat_var_log_softmax": "src/autodiff.c",
    "boat_var_sum": "src/autodiff.c",

    # ops/arithmetic.c symbols
    "boat_relu": "src/ops/arithmetic.c",

    # layers/ symbols (dense, conv, pool, flatten)
    "boat_dense_layer_create": "src/layers/dense.c",
    "boat_dense_layer_free": "src/layers/dense.c",
    "boat_dense_layer_forward": "src/layers/dense.c",
    "boat_dense_layer_get_bias": "src/layers/dense.c",
    "boat_dense_layer_get_grad_bias": "src/layers/dense.c",
    "boat_dense_layer_get_grad_weight": "src/layers/dense.c",
    "boat_dense_layer_get_weight": "src/layers/dense.c",
    "boat_dense_layer_set_bias": "src/layers/dense.c",
    "boat_dense_layer_set_weight": "src/layers/dense.c",

    "boat_conv_layer_create": "src/layers/conv.c",
    "boat_conv_layer_forward": "src/layers/conv.c",
    "boat_conv_layer_free": "src/layers/conv.c",
    "boat_conv_layer_get_bias": "src/layers/conv.c",
    "boat_conv_layer_get_grad_bias": "src/layers/conv.c",
    "boat_conv_layer_get_grad_weight": "src/layers/conv.c",
    "boat_conv_layer_get_weight": "src/layers/conv.c",
    "boat_conv_layer_set_bias": "src/layers/conv.c",
    "boat_conv_layer_set_weight": "src/layers/conv.c",

    "boat_pool_layer_create": "src/layers/pool.c",
    "boat_pool_layer_forward": "src/layers/pool.c",
    "boat_pool_layer_free": "src/layers/pool.c",

    "boat_flatten_layer_create": "src/layers/flatten.c",
    "boat_flatten_layer_forward": "src/layers/flatten.c",
    "boat_flatten_layer_free": "src/layers/flatten.c",

    # optimizers/ symbols
    "boat_adam_optimizer_create": "src/optimizers/adam.c",
    "boat_optimizer_add_parameter": "src/optimizers/optimizer_common.c",
    "boat_optimizer_free": "src/optimizers/optimizer_common.c",
    "boat_optimizer_get_learning_rate": "src/optimizers/optimizer_common.c",
    "boat_optimizer_set_learning_rate": "src/optimizers/optimizer_common.c",
    "boat_optimizer_step": "src/optimizers/optimizer_common.c",
    "boat_optimizer_zero_grad": "src/optimizers/optimizer_common.c",

    # schedulers/ symbols
    "boat_cosine_annealing_scheduler_create": "src/schedulers/cosine_annealing.c",
    "boat_scheduler_free": "src/schedulers/scheduler_common.c",
    "boat_scheduler_step": "src/schedulers/scheduler_common.c",
    "boat_scheduler_update_optimizer": "src/schedulers/scheduler_common.c",

    # memory.c symbols
    "boat_memory_get_stats": "src/core/memory.c",
    "boat_memory_reset_stats": "src/core/memory.c",
}

def add_boat_api_to_file(filename, function_name):
    """Add BOAT_API macro before function definition if missing."""
    if not os.path.exists(filename):
        print(f"Warning: File {filename} does not exist")
        return False

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Pattern to match function definition
    # Match lines like: return_type function_name(parameters) {
    # where return_type may include qualifiers like 'static', 'BOAT_API', etc.
    pattern = rf'^\s*(?:static\s+)?(?:BOAT_API\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*\s+)*\*?\s*{function_name}\s*\('

    modified = False
    for i, line in enumerate(lines):
        if re.search(pattern, line) and 'BOAT_API' not in line:
            # Check if line starts with 'static' - we shouldn't add BOAT_API to static functions
            if line.strip().startswith('static'):
                print(f"  Skipping static function {function_name} at line {i+1}")
                continue

            # Add BOAT_API before return type
            indent = len(line) - len(line.lstrip())
            new_line = line[:indent] + 'BOAT_API ' + line[indent:]
            lines[i] = new_line
            print(f"  Added BOAT_API to {function_name} at line {i+1} in {filename}")
            modified = True
            break

    if not modified:
        # Try to find function with different patterns
        # Some functions might be split across multiple lines
        simple_pattern = rf'{function_name}\s*\('
        for i, line in enumerate(lines):
            if re.search(simple_pattern, line) and 'BOAT_API' not in line:
                # Check if this looks like a function definition (not just a call)
                if (';' not in line) and (line.strip().endswith('{') or (i+1 < len(lines) and lines[i+1].strip().startswith('{'))):
                    if line.strip().startswith('static'):
                        print(f"  Skipping static function {function_name} at line {i+1}")
                        continue

                    indent = len(line) - len(line.lstrip())
                    new_line = line[:indent] + 'BOAT_API ' + line[indent:]
                    lines[i] = new_line
                    print(f"  Added BOAT_API to {function_name} at line {i+1} in {filename} (multi-line match)")
                    modified = True
                    break

    if not modified:
        print(f"  Could not find function {function_name} in {filename}")
        return False

    # Write back
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return True

def main():
    print("Fixing missing BOAT_API exports based on linker errors...")
    print(f"Total missing symbols: {len(MISSING_SYMBOLS)}")

    success_count = 0
    not_found = []

    for symbol in MISSING_SYMBOLS:
        if symbol in SYMBOL_TO_FILE:
            filename = SYMBOL_TO_FILE[symbol]
            print(f"\nProcessing {symbol} -> {filename}")
            if os.path.exists(filename):
                if add_boat_api_to_file(filename, symbol):
                    success_count += 1
            else:
                print(f"  File not found: {filename}")
                not_found.append((symbol, filename))
        else:
            print(f"\nWarning: No file mapping for symbol {symbol}")
            not_found.append((symbol, "unknown"))

    print(f"\n{'='*60}")
    print(f"Summary: Successfully updated {success_count} out of {len(MISSING_SYMBOLS)} symbols")

    if not_found:
        print(f"\nSymbols not found or files missing ({len(not_found)}):")
        for symbol, filename in not_found:
            print(f"  {symbol} -> {filename}")

    print(f"\nNext steps:")
    print(f"1. Rebuild the boat library")
    print(f"2. Try building mnist_autodiff again")
    print(f"3. Run dumpbin to verify exports")

    return 0

if __name__ == '__main__':
    sys.exit(main())