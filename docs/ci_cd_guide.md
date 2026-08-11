# Boat CI/CD Usage Guide

## Overview

Boat uses GitHub Actions for continuous integration and continuous deployment, ensuring code quality, cross-platform compatibility, and automated testing.

## CI Workflow

### Trigger Conditions
- **Push** to the main branch
- **Pull Request** to the main branch
- **Manual trigger** (workflow_dispatch)

### Workflow Files
- `.github/workflows/ci.yml` - the main CI configuration file

## Workflow Steps

### 1. Checkout Code
- Uses `actions/checkout@v4`
- Recursively initializes submodules

### 2. Install Dependencies
**Ubuntu**: Install cmake, build-essential, ccache
**macOS**: Install cmake, ccache (via Homebrew)
**Windows**: Install ccache (via Chocolatey)

### 3. Cache Configuration
- Uses `hendrikmuhs/ccache-action@v1.2` to configure and cache ccache
- Uses `runner.os` + `build_type` as the cache key

### 4. Configure CMake
```yaml
run: |
  mkdir build
  cd build
  cmake .. \
    -DBOAT_WITH_TESTS=ON \
    -DBOAT_WITH_EXAMPLES=ON \
    -DBOAT_WITH_HUGGINGFACE=ON \
    -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}
```

### 5. Build the Project
- Records the build start time
- Runs `cmake --build`
- Displays the build duration

### 6. Run Tests
- Uses `ctest --output-on-failure`
- Outputs detailed failure information

### 7. Static Analysis (Planned)
- Plans to introduce `clang-tidy` for C code quality checks (combined with `clang-analyzer`)
- CUDA code has limited static analysis tool support, relying on runtime testing and code review
- This step is not yet enabled in the current workflow

### 8. Build Examples
- Builds all examples via `-DBOAT_WITH_EXAMPLES=ON`
- Verifies that the examples compile

## Matrix Strategy

### Operating System Matrix
```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  build_type: [Release, Debug]
```

### Current Configuration
- **3 operating systems**: Ubuntu, Windows, macOS
- **2 build types**: Release, Debug
- **6 combinations in total**

### Expansion Plans
- Add a compiler matrix (GCC, Clang, MSVC)
- Add CUDA support testing
- Add sanitizer testing (AddressSanitizer, UndefinedBehaviorSanitizer)

## Performance Monitoring

### Build Time Tracking
- Records build start and end times
- Computes and displays build duration
- Monitors build time trends

### Cache Efficiency
- Displays ccache statistics
- Monitors cache hit rate
- Optimizes cache configuration

## Quality Gates

### Current Status
- The build must succeed
- All tests must pass
- Examples must build successfully

### Planned Improvements
- Set a clang-tidy warning threshold
- ✅ Code coverage reporting added (coverage job + Codecov upload)
- Add performance benchmarks

## Troubleshooting

### Common Issues

#### 1. Build Failure
- Check operating-system-specific dependencies
- Verify CMake configuration options
- Review the full build log

#### 2. Test Failure
- Check detailed test output
- Verify test data availability
- Check for cross-platform compatibility issues

#### 3. Static Analysis Warnings
- Run local clang-tidy to verify
- Refer to the [Const Usage Guide](const_usage_guide.md)
- Fix warnings incrementally

### Debugging CI
1. Enable manual workflow_dispatch triggering
2. Review detailed GitHub Actions logs
3. Reproduce the issue locally

## Running CI Steps Locally

### Installing Dependencies
```bash
# Ubuntu
sudo apt-get install -y cmake build-essential clang-tidy ccache

# macOS
brew install cmake clang-tidy ccache

# Windows
choco install cmake llvm ccache -y  # clang-tidy is part of LLVM
```

### Running the Full Flow
```bash
mkdir build
cd build
cmake .. -DBOAT_WITH_TESTS=ON -DBOAT_WITH_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure -C Release
# C static analysis
clang-tidy src/**/*.c -- -Iinclude
# or use a scan-build pass
scan-build cmake --build . --config Release
```

## Customizing the Workflow

### Adding New Steps
1. Edit `.github/workflows/ci.yml`
2. Add a new step
3. Test via workflow_dispatch

### Modifying the Matrix
```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  build_type: [Release, Debug]
  compiler: [gcc, clang]  # future extension
```

### Conditional Execution
```yaml
- name: Run GPU tests
  if: matrix.os == 'ubuntu-latest' && matrix.cuda == 'enabled'
  run: |
    # GPU-specific tests
```

## Best Practices

### 1. Fast Feedback
- Keep the workflow running fast
- Use caching to reduce build time
- Parallelize independent tasks

### 2. Reliability
- Handle transient network failures
- Set reasonable timeouts
- Provide detailed error messages

### 3. Maintainability
- Use clear step names
- Add comments explaining complex logic
- Regularly update dependency versions

### 4. Security
- Use secrets to manage sensitive information
- Regularly check dependency security
- Follow the principle of least privilege

## Future Improvement Roadmap

### Short Term (1-2 months)
- [x] Add ccache directory caching
- [ ] Expand the compiler matrix
- [ ] Add build time trend charts

### Mid Term (3-6 months)
- [x] Integrate code coverage (Codecov)
- [ ] Add performance benchmarks
- [ ] Integrate security scanning (CodeQL)

### Long Term (6+ months)
- [ ] Add release automation
- [ ] Integrate documentation generation and deployment
- [ ] Add nightly builds and testing

## Related Documentation

- [Developer Getting Started Guide](developer_getting_started.md)
- [Code Contribution Guide](contribution_guide.md)
- [Performance Optimization Guide](performance_optimization_guide.md)
- [Const Usage Guide](const_usage_guide.md)

---

*Last updated: 2026-08-05*
*Corresponding CI version: ci.yml v3.0*
