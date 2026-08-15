#!/bin/bash
# valgrind_sweep.sh - Run the whole CPU test suite (unit/integration tests plus
# fast self-contained examples) under valgrind.
#
# Usage: scripts/valgrind_sweep.sh [build_dir]   (default: build)
#
# This is the CI memory-safety gate (see .github/workflows/ci.yml, the
# "memcheck" job) and matches the local WSL2 workflow:
#   OMP_NUM_THREADS=1 valgrind --leak-check=full --error-exitcode=42 <test>
#
# Exit 0 only if every binary is valgrind-clean. Slow data-backed examples
# (cifar10, transformer) and the gracefully-skipped translator are covered by
# the regular ctest job instead.

set -u

BUILD_DIR="${1:-build}"
VG_FLAGS="--leak-check=full --error-exitcode=42 --errors-for-leak-kinds=definite,possible"
TIMEOUT_SEC=300

cd "$BUILD_DIR" || { echo "ERROR: build dir '$BUILD_DIR' not found"; exit 2; }

failures=0
count=0

run_vg() {
    local name="$1"; shift
    local envs="$1"; shift
    local t="$1"; shift
    count=$((count + 1))
    # shellcheck disable=SC2086
    if ! timeout $TIMEOUT_SEC env $envs OMP_NUM_THREADS=1 \
            valgrind $VG_FLAGS -q "$t" "$@" > /tmp/vg_$count.log 2>&1; then
        echo "MEMCHECK FAIL($?): $name"
        head -30 "/tmp/vg_$count.log"
        failures=1
    else
        echo "memcheck ok: $name"
    fi
}

# 1. Unit / integration test suite (fixtures are relative to the tests dir).
cd tests || { echo "ERROR: $BUILD_DIR/tests not found"; exit 2; }
for t in test_*; do
    [ -x "$t" ] || continue
    run_vg "$t" "" "./$t"
done
cd ..

# 2. Fast self-contained examples.
run_vg "mnist (synthetic)" "MNIST_SYNTHETIC=1" "./examples/mnist/mnist"
run_vg "scheduler_usage" "" "./examples/scheduler_usage"
run_vg "serialization" "" "./examples/serialization/serialization"
run_vg "regression" "" "./examples/regression/regression"
run_vg "needle2 --selftest" "" "./examples/needle/needle2" --selftest

if [ $failures -ne 0 ]; then
    echo "valgrind sweep FAILED ($count binaries checked)"
    exit 1
fi
echo "valgrind sweep clean ($count binaries checked)"
