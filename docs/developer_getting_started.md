# Boat Framework Developer Getting Started Guide

## Overview

Boat is a lightweight, high-performance deep learning framework written in C, supporting CPU and CUDA backends. This guide helps new developers get started with project development quickly.

## Environment Setup

### System Requirements
- **Operating System**: Linux, macOS, Windows
- **Compiler**: GCC (>= 8.0), Clang (>= 7.0), MSVC (>= 2019)
- **Build Tool**: CMake (>= 3.10)
- **Optional Dependencies**: CUDA Toolkit (>= 11.0) for GPU support

### Install Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential cppcheck ccache
```

#### macOS
```bash
brew update
brew install cmake cppcheck ccache
```

#### Windows
```bash
choco install cmake cppcheck ccache -y
```

## Getting the Code

```bash
git clone https://github.com/xiaoshaoning/boat.git
cd boat
git submodule update --init --recursive
```

## Building the Framework

### Basic Build
```bash
mkdir build
cd build
cmake .. -DBOAT_WITH_TESTS=ON -DBOAT_WITH_EXAMPLES=ON
cmake --build . --config Release
```

### Build Options
- `-DBOAT_WITH_CUDA=ON`: Enable CUDA support
- `-DBOAT_WITH_TESTS=ON`: Build the test suite
- `-DBOAT_WITH_EXAMPLES=ON`: Build example programs
- `-DBOAT_WITH_ONNX=ON`: Enable ONNX support (bundled protobuf parser)

### Install
```bash
cmake --install .
```

## Running Tests

```bash
cd build
ctest --output-on-failure -C Release
```

## Running Examples

### MNIST Handwritten Digit Recognition
```bash
./build/examples/mnist/mnist --help
```

## Project Structure

```
boat/
├── include/          # public headers
├── src/              # source code
├── examples/         # example programs
├── tests/            # test code
├── docs/             # documentation
└── .github/          # CI/CD configuration
```

## Code Style

- **Naming Convention**: snake_case (functions, variables, types)
- **Indentation**: 4 spaces
- **Line Width**: 100 characters maximum
- **Comments**: English only, no Chinese characters
- **Header Guards**: `#ifndef BOAT_FILENAME_H`

For detailed code style, please refer to CLAUDE.md.

## Debugging and Development

### Enabling Debug Mode
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBOAT_DEBUG=1
```

### Static Analysis
```bash
cppcheck --enable=warning,style --suppress=missingInclude -I include src
```

### Memory Checking
```bash
valgrind --leak-check=full ./build/tests/test_phase1
```

> Valgrind can only run on Linux/macOS. Windows developers should use WSL2 for
> memory checks (see `docs/WSL2-Valgrind-Guide.md` for the full method and results); on native
> Windows, AddressSanitizer can be used instead (`-DBOAT_WITH_ASAN=ON`).

## Contributing Code

1. Fork the repository and create a feature branch
2. Follow the code style guide
3. Add unit tests
4. Make sure all tests pass
5. Submit a Pull Request

For the detailed contribution process, please refer to the [Contribution Guide](contribution_guide.md).

## Getting Help

- **Issue Tracking**: GitHub Issues
- **Code Review**: GitHub Pull Requests
- **Documentation**: The `docs/` directory

## Next Steps

- Read the [Code Contribution Guide](contribution_guide.md) to learn the detailed process
- Check out the [CI/CD Guide](ci_cd_guide.md) to learn about automated testing
- Study the [Performance Optimization Guide](performance_optimization_guide.md) to improve code efficiency
- Refer to the [Const Usage Guide](const_usage_guide.md) to ensure code quality

---

*Last updated: 2026-03-01*
