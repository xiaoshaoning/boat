# Boat 框架开发者入门指南

## 概述

Boat 是一个轻量级、高性能的深度学习框架，使用 C 语言编写，支持 CPU 和 CUDA 后端。本指南帮助新开发者快速上手项目开发。

## 环境准备

### 系统要求
- **操作系统**: Linux, macOS, Windows
- **编译器**: GCC (>= 8.0), Clang (>= 7.0), MSVC (>= 2019)
- **构建工具**: CMake (>= 3.10)
- **可选依赖**: CUDA Toolkit (>= 11.0) 用于 GPU 支持

### 安装依赖

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

## 获取代码

```bash
git clone https://github.com/your-org/boat.git
cd boat
git submodule update --init --recursive
```

## 构建框架

### 基本构建
```bash
mkdir build
cd build
cmake .. -DBOAT_WITH_TESTS=ON -DBOAT_WITH_EXAMPLES=ON
cmake --build . --config Release
```

### 构建选项
- `-DBOAT_WITH_CUDA=ON`: 启用 CUDA 支持
- `-DBOAT_WITH_TESTS=ON`: 构建测试套件
- `-DBOAT_WITH_EXAMPLES=ON`: 构建示例程序
- `-DBOAT_WITH_ONNX=ON`: 启用 ONNX 支持 (需要 protobuf)

### 安装
```bash
cmake --install .
```

## 运行测试

```bash
cd build
ctest --output-on-failure -C Release
```

## 运行示例

### MNIST 手写数字识别
```bash
./build/examples/mnist/mnist --help
```

## 项目结构

```
boat/
├── include/          # 公共头文件
├── src/              # 源代码
├── examples/         # 示例程序
├── tests/            # 测试代码
├── docs/             # 文档
└── .github/          # CI/CD 配置
```

## 代码风格

- **命名约定**: snake_case (函数、变量、类型)
- **缩进**: 4 个空格
- **行宽**: 最大 100 字符
- **注释**: 全英文，无中文字符
- **头文件保护**: `#ifndef BOAT_FILENAME_H`

详细代码风格请参考 CLAUDE.md。

## 调试与开发

### 启用调试模式
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBOAT_DEBUG=ON
```

### 静态分析
```bash
cppcheck --enable=warning,style --suppress=missingInclude -I include src
```

### 内存检查
```bash
valgrind --leak-check=full ./build/tests/test_phase1
```

## 贡献代码

1. Fork 仓库并创建特性分支
2. 遵循代码风格指南
3. 添加单元测试
4. 确保通过所有测试
5. 提交 Pull Request

详细贡献流程请参考 [贡献指南](contribution_guide.md)。

## 获取帮助

- **问题跟踪**: GitHub Issues
- **代码审查**: GitHub Pull Requests
- **文档**: `docs/` 目录

## 下一步

- 阅读 [代码贡献指南](contribution_guide.md) 了解详细流程
- 查看 [CI/CD 指南](ci_cd_guide.md) 了解自动化测试
- 学习 [性能优化指南](performance_optimization_guide.md) 提升代码效率
- 参考 [Const 使用指南](const_usage_guide.md) 确保代码质量

---

*本指南最后更新: 2026-03-01*