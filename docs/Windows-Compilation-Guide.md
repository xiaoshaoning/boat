# Windows 编译指南

本文档详细介绍了在 Windows 平台上编译 Boat 深度学习框架的步骤、注意事项和最佳实践。重点涵盖 MSVC 编译器特定行为、DLL 构建优化控制和跨平台兼容性保障。

## 目录

1. [环境准备](#环境准备)
2. [构建配置](#构建配置)
3. [MSVC 编译器特性](#msvc-编译器特性)
4. [DLL 导出最佳实践](#dll-导出最佳实践)
5. [编译器优化控制](#编译器优化控制)
6. [调试与诊断](#调试与诊断)
7. [常见问题解决](#常见问题解决)
8. [跨平台注意事项](#跨平台注意事项)

## 环境准备

### 必备工具

1. **Visual Studio 2022 或更高版本**
   - 包含 MSVC 编译器工具链
   - 推荐安装 "使用 C++ 的桌面开发" 工作负载
   - 确保 Windows SDK 版本 10.0.19041.0 或更高

2. **CMake 3.20 或更高版本**
   - 从 [cmake.org](https://cmake.org/download/) 下载
   - 添加 CMake 到系统 PATH

3. **Git**
   - 用于获取源代码
   - 推荐使用 Git for Windows

4. **可选：Windows Terminal**
   - 提供更好的命令行体验

### 环境验证

```bash
# Verify MSVC compiler
cl.exe

# Verify CMake
cmake --version

# Verify Git
git --version
```

## 构建配置

### 基本构建步骤

```bash
# 1. Clone repository
git clone https://github.com/your-org/boat.git
cd boat

# 2. Create build directory
mkdir build
cd build

# 3. Configure CMake (shared library build)
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release

# 4. Build project
cmake --build . --config Release
```

### 关键 CMake 选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `BOAT_BUILD_SHARED` | 构建动态库 (DLL)；默认构建静态库 | `OFF` |
| `BOAT_WITH_TESTS` | 构建测试 | `OFF` |
| `BOAT_WITH_EXAMPLES` | 构建示例 | `OFF` |
| `CMAKE_BUILD_TYPE` | 构建类型 (Debug/Release/RelWithDebInfo) | 未设置 |
| `CMAKE_INSTALL_PREFIX` | 安装目录 | `C:/Program Files/boat` |

### 高级构建配置

```bash
# Debug build with symbol information
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Debug

# Release build with debug info
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Specify generator (Visual Studio 2022)
cmake .. -G "Visual Studio 17 2022" -A x64 -DBOAT_BUILD_SHARED=ON
```

## MSVC 编译器特性

### 调用约定

Windows x64 平台使用统一的调用约定，无需显式指定。Windows x86 平台需要使用 `__stdcall`。

```c
// include/boat/export.h
#if BOAT_WINDOWS && defined(_M_X64)
    // Windows x64: default calling convention (no explicit specifier needed)
    #define BOAT_CALL
#elif BOAT_WINDOWS
    // Windows x86: use __stdcall for compatibility
    #define BOAT_CALL __stdcall
#else
    // Non-Windows platforms
    #define BOAT_CALL
#endif
```

### DLL 导出/导入

使用 `__declspec(dllexport)` 和 `__declspec(dllimport)` 控制符号可见性。

```c
// include/boat/export.h
#if BOAT_WINDOWS
    // Windows DLL export/import
    #ifdef BOAT_BUILDING_DLL
        #define BOAT_API __declspec(dllexport)
    #else
        #define BOAT_API __declspec(dllimport)
    #endif
#else
    // Non-Windows platforms (Linux, macOS, etc.)
    #if __GNUC__ >= 4
        #define BOAT_API __attribute__((visibility("default")))
    #else
        #define BOAT_API
    #endif
#endif
```

### 编译器优化行为

MSVC 编译器在 Release 模式下会进行激进优化，可能导致以下问题：

1. **函数级链接 (Function-Level Linking, /Gy)**
   - 将函数打包为 COMDAT 节
   - 允许链接器消除未引用函数
   - 可能导致简单包装函数被消除

2. **内联优化**
   - 自动内联小函数
   - 对于简单包装函数，可能导致函数体被完全优化掉

3. **全局优化 (/GL) 和链接时代码生成 (LTCG)**
   - 跨模块优化
   - 增加优化能力，但也可能引入意外行为

## DLL 导出最佳实践

### 1. 简单包装函数保护

简单包装函数（特别是仅调用另一个函数的包装器）容易被编译器优化消除。使用 `BOAT_NOINLINE` 强制保留函数体。

```c
// Dangerous: simple wrapper may be optimized away
BOAT_API boat_tensor_t* boat_norm_layer_backward(boat_norm_layer_t* layer,
                                                 const boat_tensor_t* grad_output) {
    return boat_layernorm_backward(layer, grad_output);
}

// Safe: use BOAT_NOINLINE to prevent optimization
BOAT_NOINLINE BOAT_API boat_tensor_t* boat_norm_layer_backward(boat_norm_layer_t* layer,
                                                               const boat_tensor_t* grad_output) {
    return boat_layernorm_backward(layer, grad_output);
}
```

### 2. 函数声明一致性

确保头文件声明和源文件定义一致：

```c
// include/boat/layers.h - header declaration
BOAT_API boat_tensor_t* boat_dense_layer_backward(boat_dense_layer_t* layer,
                                                  const boat_tensor_t* grad_output);

// src/layers/dense.c - source definition (correct)
BOAT_API boat_tensor_t* boat_dense_layer_backward(boat_dense_layer_t* layer,
                                                  const boat_tensor_t* grad_output) {
    // implementation
}

// src/layers/dense.c - source definition (wrong, missing BOAT_API)
boat_tensor_t* boat_dense_layer_backward(boat_dense_layer_t* layer,
                                        const boat_tensor_t* grad_output) {
    // implementation - may not be exported correctly
}
```

### 3. 调用约定统一

对所有导出函数使用 `BOAT_CALL` 宏确保跨平台一致性：

```c
// Recommended: use BOAT_CALL for correct calling convention
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_forward(boat_attention_layer_t* layer,
                                                               const boat_tensor_t* query,
                                                               const boat_tensor_t* key,
                                                               const boat_tensor_t* value,
                                                               const boat_tensor_t* attention_mask);
```

### 4. 函数复杂度阈值

对于以下类型的函数，建议添加 `BOAT_NOINLINE`：

| 函数类型 | 示例 | 风险等级 |
|----------|------|----------|
| 简单包装器 | 仅调用另一个函数 | 高 |
| 返回常量 | 返回 `NULL`、`0` 等 | 高 |
| 简单 getter/setter | 返回成员变量或参数 | 中 |
| 错误检查包装器 | 检查参数后调用实际函数 | 中 |
| 复杂函数 | 包含循环、分配、系统调用 | 低 |

## 编译器优化控制

### BOAT_NOINLINE 宏

框架提供了统一的 `BOAT_NOINLINE` 宏处理各平台 noinline 属性：

```c
// include/boat/export.h
#if defined(_MSC_VER)
    #define BOAT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    #define BOAT_NOINLINE __attribute__((noinline))
#else
    #define BOAT_NOINLINE
#endif
```

### 使用场景

1. **所有层接口包装函数**
   ```c
   BOAT_NOINLINE BOAT_API boat_tensor_t* boat_norm_layer_backward(...);
   BOAT_NOINLINE BOAT_API boat_tensor_t* boat_attention_layer_forward(...);
   ```

2. **简单工具函数**
   ```c
   BOAT_NOINLINE BOAT_API const char* boat_get_version_string(void);
   BOAT_NOINLINE BOAT_API size_t boat_get_alignment(void);
   ```

3. **初始化/清理函数**
   ```c
   BOAT_NOINLINE BOAT_API void boat_initialize(void);
   BOAT_NOINLINE BOAT_API void boat_cleanup(void);
   ```

### 编译器选项控制

在 CMake 中控制编译器优化选项：

```cmake
# CMakeLists.txt snippet
if(MSVC)
    # Disable function-level linking (prevent simple functions from being eliminated)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Gy-")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Gy-")

    # Prevent elimination of unreferenced functions
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /OPT:NOREF")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:NOREF")
endif()
```

## 调试与诊断

### DLL 导出符号验证

构建后验证 DLL 导出符号：

```bash
# Use dumpbin to check exported functions
dumpbin /exports build/Release/boat.dll

# Find specific function
dumpbin /exports build/Release/boat.dll | findstr "boat_attention_layer_backward"
```

### 调试构建配置

创建专门的调试构建配置：

```bash
# Debug DLL build
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Debug

# Build and generate PDB files
cmake --build . --config Debug
```

### 运行时诊断

在代码中添加平台特定诊断：

```c
#include <boat/export.h>

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_backward(...) {
    // platform-specific debug output
#ifdef _MSC_VER
    OutputDebugStringA("[DEBUG] boat_attention_layer_backward called\n");
#endif

    // function implementation
    // ...
}
```

## 常见问题解决

### 问题 1: DLL 导出函数缺失

**症状**: 使用 `dumpbin /exports` 检查时，预期函数未出现在导出表中。

**可能原因**:
1. 函数未使用 `BOAT_API` 修饰符
2. 函数被编译器优化消除
3. 链接器优化移除了未引用函数

**解决方案**:
1. 确保函数声明和定义都使用 `BOAT_API`
2. 为简单包装函数添加 `BOAT_NOINLINE`
3. 调整链接器选项：`/OPT:NOREF`

### 问题 2: 函数调用崩溃

**症状**: 调用 DLL 导出函数时程序崩溃。

**可能原因**:
1. 调用约定不匹配
2. 堆栈指针错位
3. 参数传递错误

**解决方案**:
1. 确保所有导出函数使用 `BOAT_CALL` 宏
2. 验证函数签名一致性
3. 使用调试器检查调用堆栈

### 问题 3: 性能测试失败

**症状**: 性能测试中函数返回 `NULL` 或无效值。

**可能原因**:
1. 函数体被编译器优化消除
2. 缓存管理问题
3. 线程同步问题

**解决方案**:
1. 添加 `BOAT_NOINLINE` 防止优化
2. 添加调试输出验证函数执行
3. 检查缓存有效性

### 问题 4: 跨平台行为不一致

**症状**: 代码在 Windows 上失败，但在 Linux/macOS 上正常工作。

**可能原因**:
1. 平台特定编译器优化差异
2. DLL 与共享库机制不同
3. 调用约定差异

**解决方案**:
1. 使用统一的宏处理平台差异
2. 在各平台上运行完整测试套件
3. 实现平台兼容性测试

## 跨平台注意事项

### 宏定义兼容性

确保所有平台特定代码通过宏定义处理：

```c
// Wrong: directly using platform-specific syntax
#ifdef _MSC_VER
__declspec(noinline)
#endif
void my_function();

// Correct: use framework macro
BOAT_NOINLINE void my_function();
```

### 构建系统兼容性

CMake 配置应处理所有平台差异：

```cmake
# Platform-specific compiler options
if(MSVC)
    set(PLATFORM_C_FLAGS "/Gy- /OPT:NOREF")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(PLATFORM_C_FLAGS "-fno-inline-functions")
endif()

# Apply to all build targets
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PLATFORM_C_FLAGS}")
```

### 测试策略

1. **平台矩阵测试**: 在 Windows、Linux、macOS 上运行完整测试套件
2. **构建类型测试**: 测试 Debug、Release、RelWithDebInfo 配置
3. **链接类型测试**: 测试静态库和动态库构建
4. **编译器测试**: 测试不同编译器版本（MSVC、GCC、Clang）

### 持续集成

配置 CI 流水线包含：

```yaml
# GitHub Actions 示例
jobs:
  windows-build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure CMake
        run: cmake -DBOAT_BUILD_SHARED=ON -DBOAT_WITH_TESTS=ON -B build
      - name: Build
        run: cmake --build build --config Release
      - name: Test
        run: ctest --test-dir build --build-config Release

  linux-build:
    runs-on: ubuntu-latest
    # ...

  macos-build:
    runs-on: macos-latest
    # ...
```

## 总结

Windows 平台编译需要特别注意编译器优化行为和 DLL 机制。通过以下措施确保兼容性：

1. **统一使用框架宏**: `BOAT_API`, `BOAT_CALL`, `BOAT_NOINLINE`
2. **保护简单包装函数**: 防止编译器优化消除
3. **验证导出符号**: 构建后检查 DLL 导出表
4. **全面测试**: 多平台、多配置测试矩阵
5. **文档记录**: 记录平台特定行为和解决方案

遵循这些指南可确保 Boat 框架在 Windows 平台上的稳定性和可靠性，为生产环境部署提供坚实基础。

## 附录

### A. 实用命令参考

```bash
# Generate Visual Studio solution
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build specific configuration
cmake --build . --config Release --target boat

# Run tests
ctest -C Release -V

# Check DLL dependencies
dumpbin /dependents boat.dll

# Check DLL exported functions
dumpbin /exports boat.dll > exports.txt
```

### B. 推荐开发工具

1. **Visual Studio 2022**: 集成开发环境
2. **CMake GUI**: 图形化配置工具
3. **Dependencies** (formerly Dependency Walker): DLL 分析工具
4. **Process Monitor**: 系统监控工具
5. **DebugView**: 系统调试输出查看器

### C. 参考资料

1. [Microsoft C/C++ 文档](https://docs.microsoft.com/cpp/)
2. [CMake 文档](https://cmake.org/documentation/)
3. [Windows DLL 最佳实践](https://docs.microsoft.com/windows/win32/dlls/dynamic-link-library-best-practices)
4. [MSVC 编译器选项](https://docs.microsoft.com/cpp/build/reference/compiler-options-listed-alphabetically)

---

*最后更新: 2026-02-24*
*文档版本: 1.0*