#!/usr/bin/env python3
import re
import sys

def clean_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    new_lines = []
    while i < len(lines):
        line = lines[i]
        # Skip fprintf(stderr, "[DEBUG]" lines and their continuations
        if 'fprintf(stderr, "[DEBUG]"' in line:
            # Skip this line and any following lines that are just parameters
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip printf("[DEBUG]" lines
        if 'printf("[DEBUG]"' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip fprintf(stderr, "DEBUG" lines
        if 'fprintf(stderr, "DEBUG"' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip printf("DEBUG... (generic) lines
        if 'printf("DEBUG' in line and not 'printf("DEBUG"' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip fprintf(stderr, "DEBUG... (generic) lines
        if 'fprintf(stderr, "DEBUG' in line and not 'fprintf(stderr, "DEBUG"' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip printf("DEBUG ..." lines (with space after DEBUG)
        if 'printf("DEBUG ' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip fprintf(stderr, "DEBUG ..." lines
        if 'fprintf(stderr, "DEBUG ' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip printf("DEBUG" lines
        if 'printf("DEBUG"' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip fflush(stderr); lines
        if 'fflush(stderr);' in line:
            i += 1
            continue
        # Skip FILE* f = fopen("..._called.txt" lines and their block
        if 'fopen("' in line and '_called.txt"' in line:
            # Skip this line and next few lines until closing brace
            i += 1
            brace_count = 0
            while i < len(lines):
                if '{' in lines[i]:
                    brace_count += 1
                if '}' in lines[i]:
                    brace_count -= 1
                    if brace_count == 0:
                        i += 1
                        break
                i += 1
            continue
        # Skip printf("[INFO] ..." lines (with space after INFO)
        if 'printf("[INFO] ' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip fprintf(stderr, "[INFO] ..." lines
        if 'fprintf(stderr, "[INFO] ' in line:
            i += 1
            while i < len(lines) and lines[i].strip().endswith(');'):
                i += 1
            continue
        # Skip printf("[INFO]" lines
        if 'printf("[INFO]"' in line:
            i += 1
            continue
        # Skip fflush(stdout); lines that follow debug prints
        if 'fflush(stdout);' in line:
            i += 1
            continue
        new_lines.append(line)
        i += 1

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Cleaned {filename}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean_debug.py <file>")
        sys.exit(1)
    for file in sys.argv[1:]:
        clean_file(file)