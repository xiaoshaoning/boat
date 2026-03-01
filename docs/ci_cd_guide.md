# Boat CI/CD 使用指南

## 概述

Boat 使用 GitHub Actions 进行持续集成和持续部署，确保代码质量、跨平台兼容性和自动化测试。

## CI 工作流

### 触发条件
- **推送**到 main 分支
- **Pull Request** 到 main 分支
- **手动触发** (workflow_dispatch)

### 工作流文件
- `.github/workflows/ci.yml` - 主要 CI 配置文件

## 工作流步骤

### 1. 代码检出
- 使用 `actions/checkout@v3`
- 递归初始化子模块

### 2. 依赖安装
**Ubuntu**: 安装 cmake, build-essential, cppcheck, ccache
**macOS**: 安装 cmake, cppcheck, ccache (通过 Homebrew)
**Windows**: 安装 cmake, cppcheck, ccache (通过 Chocolatey)

### 3. 缓存配置
- 使用 ccache 加速编译
- 缓存 ccache 目录以提高后续构建速度
- 设置最大缓存大小: 1GB

### 4. 配置 CMake
```yaml
env:
  CMAKE_C_COMPILER_LAUNCHER: ccache
  CMAKE_CXX_COMPILER_LAUNCHER: ccache
run: |
  mkdir build
  cd build
  cmake .. -DBOAT_WITH_TESTS=ON -DBOAT_WITH_EXAMPLES=ON -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}
```

### 5. 构建项目
- 记录构建开始时间
- 执行 `cmake --build`
- 显示构建持续时间

### 6. 运行测试
- 使用 `ctest --output-on-failure`
- 输出详细失败信息

### 7. 静态分析
- 使用 cppcheck 进行代码质量检查
- 启用警告、样式、性能、可移植性检查
- 目前不因警告而失败 (使用 `|| true`)

### 8. 示例验证
- 构建并运行 MNIST 示例
- 验证基本功能正常

## 矩阵策略

### 操作系统矩阵
```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  build_type: [Release, Debug]
```

### 当前配置
- **3 个操作系统**: Ubuntu, Windows, macOS
- **2 个构建类型**: Release, Debug
- **总计 6 个组合**

### 扩展计划
- 添加编译器矩阵 (GCC, Clang, MSVC)
- 添加 CUDA 支持测试
- 添加 sanitizer 测试 (AddressSanitizer, UndefinedBehaviorSanitizer)

## 性能监控

### 构建时间跟踪
- 记录构建开始和结束时间
- 计算并显示构建持续时间
- 监控构建时间变化趋势

### 缓存效率
- 显示 ccache 统计信息
- 监控缓存命中率
- 优化缓存配置

## 质量门禁

### 当前状态
- 构建必须成功
- 所有测试必须通过
- 示例必须构建成功

### 计划改进
- 设置 cppcheck 警告阈值
- 添加代码覆盖率要求
- 添加性能基准测试

## 故障排除

### 常见问题

#### 1. 构建失败
- 检查操作系统特定依赖
- 验证 CMake 配置选项
- 查看完整构建日志

#### 2. 测试失败
- 检查测试输出详细信息
- 验证测试数据可用性
- 检查跨平台兼容性问题

#### 3. 静态分析警告
- 运行本地 cppcheck 验证
- 参考 [Const 使用指南](const_usage_guide.md)
- 逐步修复警告

### 调试 CI
1. 启用 workflow_dispatch 手动触发
2. 查看 GitHub Actions 详细日志
3. 在本地复现问题

## 本地运行 CI 步骤

### 安装依赖
```bash
# Ubuntu
sudo apt-get install -y cmake build-essential cppcheck ccache

# macOS
brew install cmake cppcheck ccache

# Windows
choco install cmake cppcheck ccache -y
```

### 运行完整流程
```bash
mkdir build
cd build
cmake .. -DBOAT_WITH_TESTS=ON -DBOAT_WITH_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure -C Release
cppcheck --enable=warning,style,performance,portability --suppress=missingInclude -I include src
```

## 自定义工作流

### 添加新步骤
1. 编辑 `.github/workflows/ci.yml`
2. 添加新的 step
3. 测试通过 workflow_dispatch

### 修改矩阵
```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  build_type: [Release, Debug]
  compiler: [gcc, clang]  # 未来扩展
```

### 条件执行
```yaml
- name: Run GPU tests
  if: matrix.os == 'ubuntu-latest' && matrix.cuda == 'enabled'
  run: |
    # GPU 特定测试
```

## 最佳实践

### 1. 快速反馈
- 保持工作流快速运行
- 使用缓存减少构建时间
- 并行化独立任务

### 2. 可靠性
- 处理临时网络故障
- 设置合理超时时间
- 提供详细错误信息

### 3. 可维护性
- 使用清晰的步骤名称
- 添加注释说明复杂逻辑
- 定期更新依赖版本

### 4. 安全性
- 使用 secrets 管理敏感信息
- 定期检查依赖安全性
- 遵循最小权限原则

## 未来改进路线图

### 短期 (1-2 个月)
- [ ] 添加 ccache 目录缓存
- [ ] 扩展编译器矩阵
- [ ] 添加构建时间趋势图

### 中期 (3-6 个月)
- [ ] 集成代码覆盖率 (Coveralls/Codecov)
- [ ] 添加性能基准测试
- [ ] 集成安全扫描 (CodeQL)

### 长期 (6+ 个月)
- [ ] 添加发布自动化
- [ ] 集成文档生成和部署
- [ ] 添加 nightly 构建和测试

## 相关文档

- [开发者入门指南](developer_getting_started.md)
- [代码贡献指南](contribution_guide.md)
- [性能优化指南](performance_optimization_guide.md)
- [Const 使用指南](const_usage_guide.md)

---

*本指南最后更新: 2026-03-01*
*对应 CI 版本: ci.yml v2.0*