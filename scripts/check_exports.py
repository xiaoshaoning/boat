#!/usr/bin/env python3
"""
Script to check which functions already have BOAT_API and which are missing.
"""

import os
import re
import sys

# List of missing symbols from linker errors
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
]

# Map symbols to likely source files
SYMBOL_TO_FILE = {
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
}

def check_function_export_status(filename, function_name):
    """Check if a function already has BOAT_API macro."""
    if not os.path.exists(filename):
        return "file_not_found"

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match function definition
    # Looking for: BOAT_API return_type function_name OR just return_type function_name
    pattern1 = rf'BOAT_API\s+[a-zA-Z_][a-zA-Z0-9_]*\s+\*?\s*{function_name}\s*\('
    pattern2 = rf'^\s*(?!BOAT_API)(?:[a-zA-Z_][a-zA-Z0-9_]*\s+)+\*?\s*{function_name}\s*\('

    if re.search(pattern1, content, re.MULTILINE):
        return "has_boat_api"
    elif re.search(pattern2, content, re.MULTILINE):
        return "missing_boat_api"
    else:
        # Try to find function with simpler pattern (might be split across lines)
        simple_pattern = rf'{function_name}\s*\('
        if re.search(simple_pattern, content):
            # Check if it's a definition (not a call) by looking for { after it
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if function_name in line and '(' in line:
                    # Check if this is a definition (has { on same or next line)
                    if '{' in line or (i+1 < len(lines) and lines[i+1].strip().startswith('{')):
                        if 'BOAT_API' in line:
                            return "has_boat_api"
                        else:
                            return "missing_boat_api"
        return "not_found"

def main():
    print("Checking BOAT_API export status for missing symbols...")
    print(f"Total symbols to check: {len(MISSING_SYMBOLS)}")

    results = {
        "has_boat_api": [],
        "missing_boat_api": [],
        "not_found": [],
        "file_not_found": []
    }

    for symbol in MISSING_SYMBOLS:
        if symbol in SYMBOL_TO_FILE:
            filename = SYMBOL_TO_FILE[symbol]
            status = check_function_export_status(filename, symbol)
            results[status].append((symbol, filename))
        else:
            results["not_found"].append((symbol, "unknown"))

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Has BOAT_API: {len(results['has_boat_api'])}")
    print(f"  Missing BOAT_API: {len(results['missing_boat_api'])}")
    print(f"  Not found in file: {len(results['not_found'])}")
    print(f"  File not found: {len(results['file_not_found'])}")

    if results["missing_boat_api"]:
        print(f"\nFunctions missing BOAT_API ({len(results['missing_boat_api'])}):")
        for symbol, filename in results["missing_boat_api"]:
            print(f"  {symbol} -> {filename}")

    if results["has_boat_api"]:
        print(f"\nFunctions already have BOAT_API ({len(results['has_boat_api'])}):")
        for symbol, filename in results["has_boat_api"][:10]:  # Show first 10
            print(f"  {symbol} -> {filename}")
        if len(results["has_boat_api"]) > 10:
            print(f"  ... and {len(results['has_boat_api']) - 10} more")

    if results["not_found"]:
        print(f"\nSymbols not found in mapped files ({len(results['not_found'])}):")
        for symbol, filename in results["not_found"]:
            print(f"  {symbol} -> {filename}")

    if results["file_not_found"]:
        print(f"\nFiles not found ({len(results['file_not_found'])}):")
        for symbol, filename in results["file_not_found"]:
            print(f"  {symbol} -> {filename}")

    # Create fix commands for missing BOAT_API functions
    if results["missing_boat_api"]:
        print(f"\n{'='*60}")
        print("Suggested fix commands:")
        for symbol, filename in results["missing_boat_api"]:
            print(f"# To fix {symbol}:")
            print(f"# Check {filename} for function definition and add BOAT_API before return type")

    return 0

if __name__ == '__main__':
    sys.exit(main())