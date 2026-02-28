# 重新构建纯CPU版本 Boat DLL 指南

## 问题概述
当前 `boat.dll` 链接了包含CUDA依赖的PyTorch版本，导致需要 `libiomp5md.dll` 和 `cupti64_2025.3.0.dll`。
本指南将指导您重新构建一个纯CPU版本的 `boat.dll`。

## 环境检查
您的系统已安装以下必要工具：
- ✅ Visual Studio 2022 (版本18)
- ✅ MSBuild 工具链
- ✅ PyTorch LibTorch 库（多个版本可用）

## 步骤1：选择纯CPU PyTorch版本

您有两个PyTorch版本可选：

### 选项A: 使用现有的较小版本 (115MB)
```bash
# 路径: D:\codes\boat\external\libtorch-shared-with-deps-latest\libtorch
# 大小: 115MB (可能是不含CUDA的版本)
set BOAT_PYTORCH_PATH="D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch"
```

### 选项B: 下载官方纯CPU LibTorch (推荐)
1. 访问 [PyTorch官网](https://pytorch.org/get-started/locally/)
2. 选择配置：
   - PyTorch Build: Stable (1.13.1)
   - Your OS: Windows
   - Package: LibTorch
   - Language: C++/Java
   - Compute Platform: CPU
3. 下载 **Windows CPU-only** 版本
4. 解压到 `D:\codes\boat\external\libtorch-cpu-only\`
5. 设置路径：
```bash
set BOAT_PYTORCH_PATH="D:/codes/boat/external/libtorch-cpu-only"
```

## 步骤2：清理现有构建

```bash
cd D:\codes\boat

# 删除所有现有构建目录
rmdir /s /q build_test_pytorch3
rmdir /s /q build_mnist_test
rmdir /s /q build_fixed

# 创建新的构建目录
mkdir build_cpu_only
cd build_cpu_only
```

## 步骤3：配置CMake

### 方法A: 使用CMake命令行 (如果已安装CMake)
```bash
# 如果之前安装了CMake
cmake -DBOAT_WITH_PYTORCH=ON ^
      -DBOAT_PYTORCH_PATH="D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch" ^
      -DBOAT_WITH_CUDA=OFF ^
      -DBOAT_WITH_TESTS=ON ^
      -A x64 ^
      ..
```

### 方法B: 使用Visual Studio开发者命令提示符
1. 打开 **x64 Native Tools Command Prompt for VS 2022**
2. 导航到项目目录：
```cmd
cd D:\codes\boat
mkdir build_cpu_only
cd build_cpu_only
```
3. 运行CMake配置：
```cmd
cmake -DBOAT_WITH_PYTORCH=ON ^
      -DBOAT_PYTORCH_PATH="D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch" ^
      -DBOAT_WITH_CUDA=OFF ^
      -DBOAT_WITH_TESTS=ON ^
      -G "Visual Studio 17 2022" ^
      -A x64 ^
      ..
```

### 方法C: 使用CMake-GUI (可视化工具)
1. 打开 CMake-GUI
2. 设置源路径: `D:/codes/boat`
3. 设置构建路径: `D:/codes/boat/build_cpu_only`
4. 点击 **Configure**
5. 选择生成器: **Visual Studio 17 2022**
6. 选择平台: **x64**
7. 设置以下选项：
   - `BOAT_WITH_PYTORCH`: ON
   - `BOAT_PYTORCH_PATH`: `D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch`
   - `BOAT_WITH_CUDA`: OFF
   - `BOAT_WITH_TESTS`: ON
8. 点击 **Generate**

## 步骤4：构建 boat.dll

### 方法A: 使用Visual Studio IDE
1. 打开 `build_cpu_only\boat.sln`
2. 在解决方案资源管理器中右键点击 **boat** 项目
3. 选择 **生成** → **重新生成**
4. 等待构建完成

### 方法B: 使用命令行
```bash
cd D:\codes\boat\build_cpu_only
cmake --build . --config Release
```

## 步骤5：验证构建

### 检查生成的DLL
```bash
ls -la build_cpu_only/Release/boat.dll
# 文件大小应在 100-200KB 左右

# 检查依赖关系
objdump -p build_cpu_only/Release/boat.dll | grep "DLL Name"
# 应该只显示 torch_cpu.dll, c10.dll, Windows系统DLL等
# 不应该有 libiomp5md.dll 或 cupti64*.dll
```

### 检查PyTorch依赖
```bash
# 检查torch_cpu.dll的依赖
objdump -p external/libtorch-shared-with-deps-latest/libtorch/lib/torch_cpu.dll | grep "DLL Name"
# 确认不依赖CUPTI库
```

## 步骤6：测试MNIST CNN模型

### 准备测试文件
```bash
cd D:\codes\boat\external\mnist-cnn-digit-classifier

# 清理旧的DLL
del boat.dll

# 复制新的boat.dll
cp ..\..\build_cpu_only\Release\boat.dll .

# 复制PyTorch依赖库
cp "D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch/lib/torch_cpu.dll" .
cp "D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch/lib/c10.dll" .
cp "D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch/lib/torch.dll" .
cp "D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch/lib/torch_global_deps.dll" .

# 检查是否还需要其他依赖
# 复制Intel OpenMP库（如果需要）
cp "D:/codes/boat/external/libtorch-shared-with-deps-latest/libtorch/lib/libiomp5md.dll" .
```

### 重新编译测试程序
```bash
# 使用新的库路径
gcc -std=c11 -Wall -I../../include -L../../build_cpu_only/Release ^
    test_boat_load.c -lboat -lm -o test_mnist_cpu.exe
```

### 运行测试
```bash
./test_mnist_cpu.exe
```

预期输出：
```
Testing Boat PyTorch model loading for MNIST CNN...
Model loaded successfully!
Running forward pass...
Output shape: [1, 10]
Output elements: 10
First 10 output values: ...
Test completed successfully!
```

## 故障排除

### 问题1: CMake找不到Torch
```
CMake Error at CMakeLists.txt:130 (find_package):
  Could not find a package configuration file provided by "Torch"
```

**解决方案**：
1. 确保 `BOAT_PYTORCH_PATH` 指向正确的LibTorch目录
2. 检查目录中是否有 `share/cmake/Torch/TorchConfig.cmake`
3. 或者尝试手动指定库路径

### 问题2: 链接错误
```
undefined reference to `__imp_boat_tensor_create'
```

**解决方案**：
1. 确保 `include/boat/export.h` 正确包含 `BOAT_API` 宏
2. 检查头文件修改已提交并生效
3. 重新生成构建文件

### 问题3: 运行时缺少DLL
```
The program can't start because XXX.dll is missing
```

**解决方案**：
1. 使用 `objdump -p boat.dll | grep "DLL Name"` 检查依赖
2. 将所有必需的DLL复制到测试目录
3. 确保使用相同架构的DLL（都是x64）

### 问题4: PyTorch版本不兼容
```
Failed to load model: Unsupported PyTorch version
```

**解决方案**：
1. 确保PyTorch版本与模型兼容
2. 检查模型是否使用正确的TorchScript版本保存
3. 尝试重新转换模型

## 成功标准

1. ✅ `boat.dll` 构建成功（无错误）
2. ✅ `boat.dll` 依赖检查显示无CUDA相关库
3. ✅ 测试程序编译成功
4. ✅ MNIST模型加载成功
5. ✅ 前向传播执行成功
6. ✅ 输出形状正确 [1, 10]

## 后续步骤

一旦纯CPU版本构建成功：

1. **提交代码**：将构建配置更新到代码库
2. **更新文档**：记录构建过程和最佳实践
3. **创建持续集成**：确保未来构建都使用纯CPU配置
4. **扩展测试**：添加更多模型测试用例
5. **性能优化**：针对CPU进行性能调优

## 联系支持

如果遇到问题，请提供：
1. CMake配置输出
2. 构建错误日志
3. `objdump -p boat.dll` 的输出
4. 测试程序的完整错误信息

---

**最后更新**: 2026-02-22
**版本**: 1.0
**状态**: 草稿（等待验证）