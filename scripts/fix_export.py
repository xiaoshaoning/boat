#!/usr/bin/env python3
"""
Script to add BOAT_API macro to function definitions in C files.
Based on the list of functions missing BOAT_API from exploration agent.
"""

import os
import re
import sys

# Function list: (filename, function_name)
# Note: This list may be incomplete; should be updated as needed.
FUNCTIONS_TO_FIX = [
    # tensor.c
    ("src/core/tensor.c", "boat_tensor_free"),
    ("src/core/tensor.c", "boat_tensor_ref"),
    ("src/core/tensor.c", "boat_tensor_device"),
    ("src/core/tensor.c", "boat_dtype_name"),
    ("src/core/tensor.c", "boat_tensor_reshape"),

    # autodiff.c
    ("src/autodiff.c", "boat_variable_data"),
    ("src/autodiff.c", "boat_variable_grad"),
    ("src/autodiff.c", "boat_var_add"),
    ("src/autodiff.c", "boat_var_sub"),
    ("src/autodiff.c", "boat_var_div"),
    ("src/autodiff.c", "boat_var_matmul"),
    ("src/autodiff.c", "boat_var_dot"),
    ("src/autodiff.c", "boat_var_relu"),
    ("src/autodiff.c", "boat_var_sigmoid"),
    ("src/autodiff.c", "boat_var_tanh"),
    ("src/autodiff.c", "boat_var_softmax"),
    ("src/autodiff.c", "boat_var_flatten"),
    ("src/autodiff.c", "boat_var_log_softmax"),
    ("src/autodiff.c", "boat_var_conv"),
    ("src/autodiff.c", "boat_var_pool"),
    ("src/autodiff.c", "boat_var_dense"),
    ("src/autodiff.c", "boat_var_attention"),
    ("src/autodiff.c", "boat_var_sum"),
    ("src/autodiff.c", "boat_var_mean"),
    ("src/autodiff.c", "boat_var_max"),
    ("src/autodiff.c", "boat_var_min"),
    ("src/autodiff.c", "boat_autodiff_context_create"),
    ("src/autodiff.c", "boat_autodiff_context_get_graph"),
    ("src/autodiff.c", "boat_autodiff_get_current_context"),

    # optimizers/optimizer_common.c
    ("src/optimizers/optimizer_common.c", "boat_optimizer_step"),
    ("src/optimizers/optimizer_common.c", "boat_optimizer_zero_grad"),
    ("src/optimizers/optimizer_common.c", "boat_optimizer_free"),
    ("src/optimizers/optimizer_common.c", "boat_optimizer_add_parameter"),

    # model/model.c
    ("src/model/model.c", "boat_model_create"),
    ("src/model/model.c", "boat_model_create_with_graph"),
    ("src/model/model.c", "boat_model_graph"),
    ("src/model/model.c", "boat_model_forward"),
    ("src/model/model.c", "boat_model_backward"),
    ("src/model/model.c", "boat_model_load"),

    # graph/graph.c
    ("src/graph/graph.c", "boat_graph_copy"),
    ("src/graph/graph.c", "boat_graph_get_node_at_index"),
    ("src/graph/graph.c", "boat_graph_get_edge_at_index"),

    # ops/arithmetic.c
    ("src/ops/arithmetic.c", "boat_mod"),
    ("src/ops/arithmetic.c", "boat_pow_scalar"),
    ("src/ops/arithmetic.c", "boat_broadcast_to"),
    ("src/ops/arithmetic.c", "boat_sum"),
]

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
    # We'll search for function_name followed by '('
    pattern = rf'^\s*(?:static\s+)?(?:BOAT_API\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*\s+)*\*?\s*{function_name}\s*\('

    modified = False
    for i, line in enumerate(lines):
        if re.search(pattern, line) and 'BOAT_API' not in line:
            # Check if line starts with 'static' - we shouldn't add BOAT_API to static functions
            if line.strip().startswith('static'):
                print(f"  Skipping static function {function_name} at line {i+1}")
                continue

            # Add BOAT_API before return type
            # Find where the return type ends and function name begins
            # Simple approach: insert 'BOAT_API ' at the beginning of the line after any leading whitespace
            indent = len(line) - len(line.lstrip())
            new_line = line[:indent] + 'BOAT_API ' + line[indent:]
            lines[i] = new_line
            print(f"  Added BOAT_API to {function_name} at line {i+1}")
            modified = True
            break

    if not modified:
        print(f"  Could not find function {function_name} in {filename}")
        # Try alternative search: maybe function signature spans multiple lines
        # For now, just report
        return False

    # Write back
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return True

def main():
    print("Adding BOAT_API to missing function definitions...")
    success_count = 0
    total = len(FUNCTIONS_TO_FIX)

    for filename, func_name in FUNCTIONS_TO_FIX:
        print(f"Processing {func_name} in {filename}")
        if add_boat_api_to_file(filename, func_name):
            success_count += 1

    print(f"\nDone. Successfully updated {success_count} out of {total} functions.")

    # Build verification step (optional)
    print("\nConsider running build to verify linking errors are resolved.")

    return 0 if success_count > 0 else 1

if __name__ == '__main__':
    sys.exit(main())