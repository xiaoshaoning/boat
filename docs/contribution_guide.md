# Boat Contribution Guide

## Overview

Welcome to contributing code to the Boat deep learning framework! This guide will help you understand the contribution process, code standards, and best practices.

## Contribution Process

### 1. Preparation
1. **Fork the repository**: Click the "Fork" button in the top-right corner of the GitHub page
2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/boat.git
   cd boat
   ```
3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/boat.git
   ```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```
**Branch naming convention**:
- `feature/` - new features
- `fix/` - bug fixes
- `docs/` - documentation updates
- `test/` - test-related changes
- `refactor/` - code refactoring

### 3. Development and Testing
1. **Implement the feature**: Follow the code style guide
2. **Add tests**: Write unit tests for the new feature
3. **Run tests locally**:
   ```bash
   mkdir build && cd build
   cmake .. -DBOAT_WITH_TESTS=ON
   cmake --build .
   ctest --output-on-failure
   ```
4. **Static analysis**:
   ```bash
   # C static analysis (clang-tidy, not cppcheck, which targets C++ mainly)
   clang-tidy src/**/*.c -- -Iinclude
   ```

### 4. Commit Changes
```bash
git add .
git commit -m "<type>: descriptive commit message"
```
**Commit message format**:
```
<type>: short description

Detailed description (optional)

- list the main changes
- describe the impact scope
- reference issue numbers (e.g. #123)

Type legend:
- feat: new feature
- fix: bug fix
- docs: documentation update
- style: code style changes (no functional impact)
- refactor: code refactoring
- test: test-related
- chore: build process or tooling update
```

### 5. Sync with Upstream Changes
```bash
git fetch upstream
git rebase upstream/main
```

### 6. Push Changes
```bash
git push origin feature/your-feature-name
```

### 7. Create a Pull Request
1. Visit the GitHub repository page
2. Click "New Pull Request"
3. Select your branch
4. Fill in the PR description template
5. Wait for CI to run and for code review

## Code Standards

### Naming Conventions
- **Functions**: `snake_case`
- **Variables**: `snake_case`
- **Types**: `snake_type_t`
- **Constants**: `SNAKE_CASE`
- **Files**: `snake_case.c`, `snake_case.h`

### Code Style
- **Indentation**: 4 spaces (not tabs)
- **Line width**: 100 characters maximum
- **Braces**: K&R style
- **Comments**: entirely in English, no Chinese characters
- **Encoding**: All text files (code, docs, scripts) must be UTF-8 encoded (without BOM);
  other encodings such as GBK/GB2312 must not be committed. Please follow the .editorconfig in the repository root.

### Example
```c
// function declaration
boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim,
                                  boat_dtype_t dtype, boat_device_t device);

// struct definition
struct boat_tensor_t {
    int64_t* shape;
    size_t ndim;
    boat_dtype_t dtype;
};

// constant definition
#define BOAT_MAX_DIMS 8
```

## Quality Requirements

### 1. Code Correctness
- Pass all existing tests
- Add test coverage for new features
- Handle edge cases and error conditions

### 2. Memory Safety
- No memory leaks (check with Valgrind)
- No dangling pointers
- Proper reference counting management

### 3. Performance Considerations
- Avoid unnecessary memory allocations
- Use efficient algorithms
- Consider cache friendliness

### 4. Maintainability
- Clear code structure
- Meaningful variable names
- Appropriate comments (explain why, not what)

## Testing Requirements

### Unit Tests
- Every new feature should have corresponding unit tests
- Tests should cover both normal and error cases
- Test files are named: `test_<module>.c`

### Integration Tests
- Verify interactions between modules
- Test end-to-end functionality
- Ensure backward compatibility

### Test Structure
```c
#include <boat/test.h>

TEST(test_function_name) {
    // test code
    ASSERT(condition, "error message");
    ASSERT_EQ(expected, actual);
    ASSERT_NEAR(float_expected, float_actual, epsilon);
}

int main() {
    RUN_TEST(test_function_name);
    return 0;
}
```

## Documentation Requirements

### Code Documentation
- Public APIs must have doc comments
- Complex algorithms should have explanatory comments
- Header files should describe the module's purpose

### API Documentation Example
```c
/**
 * create new tensor
 *
 * @param shape tensor shape array, length is ndim
 * @param ndim number of tensor dimensions (0 means scalar)
 * @param dtype data type
 * @param device device type (CPU/GPU)
 * @return new tensor pointer, returns NULL on failure
 */
BOAT_API boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim,
                                           boat_dtype_t dtype, boat_device_t device);
```

### User Documentation
- Update related documentation (if needed)
- Add usage examples
- Update the README (if installation or usage is affected)

## Review Process

### PR Review Standards
1. **Functional correctness**: Does the implementation meet the requirements
2. **Code quality**: Does it conform to the code standards
3. **Test coverage**: Are there sufficient tests
4. **Documentation completeness**: Are related docs updated
5. **Performance impact**: Does it affect existing performance

### Review Feedback
- Constructive criticism
- Specific improvement suggestions
- Explain the reasoning behind review decisions

### Common Review Comments
- **Need more tests**: Add test cases
- **Code style issues**: Follow the style guide
- **Missing documentation**: Add API docs
- **Performance issues**: Optimize algorithms or memory usage

## Special Contribution Types

### Bug Fixes
1. Create a minimal reproduction case
2. Identify the root cause
3. Provide a fix
4. Add regression tests

### Performance Optimization
1. Provide performance benchmarks
2. Demonstrate the optimization gains
3. Make sure functionality is not broken
4. Update related documentation

### Documentation Improvements
1. Ensure the information is accurate
2. Keep the style consistent
3. Add practical examples
4. Check that links are valid

## Tooling Support

### Development Tools
- **Code formatting**: Follow the code style in CLAUDE.md
- **Static analysis**: `clang-tidy` integration (C code; CUDA static analysis tooling is limited)
- **Build system**: CMake

### Local Checks
```bash
# run static analysis
clang-tidy src/**/*.c -- -Iinclude

# run full test suite
mkdir -p build && cd build
cmake .. -DBOAT_WITH_TESTS=ON
cmake --build .
ctest --output-on-failure
```

## Community Guidelines

### Code of Conduct
1. **Respect**: Respect all community members
2. **Inclusiveness**: Welcome contributors from diverse backgrounds
3. **Collaboration**: Collaborate actively to solve problems together
4. **Professionalism**: Keep discussions professional and technical

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Technical discussions and Q&A
- **Pull Requests**: Code contributions and reviews

## Quick Reference

### Common Commands
```bash
# setup development environment
git clone --recursive https://github.com/your-username/boat.git
cd boat
mkdir build && cd build
cmake .. -DBOAT_WITH_TESTS=ON -DCMAKE_BUILD_TYPE=Debug

# daily development loop
make                    # build
ctest -V               # run tests
clang-tidy src/*.c -- -Iinclude  # static check (C code only)

# commit changes
git add .
git commit -m "feat: add new feature"
git push origin feature/xxx
```

### Resource Links
- [Developer Getting Started Guide](developer_getting_started.md)
- [CI/CD Guide](ci_cd_guide.md)
- [Performance Optimization Guide](performance_optimization_guide.md)
- [Const Usage Guide](const_usage_guide.md)
- [Detailed Code Style Guide](CLAUDE.md)

## Troubleshooting

### Frequently Asked Questions
**Q: My PR was rejected. What should I do?**
A: Read the review comments carefully, fix the code, and resubmit. If you have questions, politely ask for clarification.

**Q: How do I add a new dependency?**
A: Add the dependency in CMakeLists.txt and update the documentation. Major dependency changes require discussion.

**Q: Tests pass on my machine but fail in CI?**
A: Check for cross-platform compatibility issues and make sure the tests do not depend on a specific environment.

**Q: When will my contribution be merged?**
A: It depends on the complexity of the PR, the review progress, and project priorities. It usually takes 1-2 weeks.

---

*Last updated: 2026-03-01*
*Suggestions for improvement are welcome!*
