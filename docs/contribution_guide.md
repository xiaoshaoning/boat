# Boat 代码贡献指南

## 概述

欢迎为 Boat 深度学习框架贡献代码！本指南将帮助您了解贡献流程、代码标准和最佳实践。

## 贡献流程

### 1. 准备工作
1. **Fork 仓库**: 点击 GitHub 页面右上角的 "Fork" 按钮
2. **克隆仓库**:
   ```bash
   git clone https://github.com/your-username/boat.git
   cd boat
   git submodule update --init --recursive
   ```
3. **设置上游远程**:
   ```bash
   git remote add upstream https://github.com/original-owner/boat.git
   ```

### 2. 创建特性分支
```bash
git checkout -b feature/your-feature-name
```
**分支命名约定**:
- `feature/` - 新功能
- `fix/` - 错误修复
- `docs/` - 文档更新
- `test/` - 测试相关
- `refactor/` - 代码重构

### 3. 开发与测试
1. **实现功能**: 遵循代码风格指南
2. **添加测试**: 为新功能编写单元测试
3. **本地测试**:
   ```bash
   mkdir build && cd build
   cmake .. -DBOAT_WITH_TESTS=ON
   cmake --build .
   ctest --output-on-failure
   ```
4. **静态分析**:
   ```bash
   cppcheck --enable=warning,style --suppress=missingInclude -I include src
   ```

### 4. 提交更改
```bash
git add .
git commit -m "类型: 描述性提交信息"
```
**提交信息格式**:
```
类型: 简要描述

详细描述（可选）

- 列出主要更改
- 说明影响范围
- 关联问题编号（如 #123）

类型说明:
- feat: 新功能
- fix: 错误修复
- docs: 文档更新
- style: 代码风格调整（不影响功能）
- refactor: 代码重构
- test: 测试相关
- chore: 构建过程或工具更新
```

### 5. 同步上游更改
```bash
git fetch upstream
git rebase upstream/main
```

### 6. 推送更改
```bash
git push origin feature/your-feature-name
```

### 7. 创建 Pull Request
1. 访问 GitHub 仓库页面
2. 点击 "New Pull Request"
3. 选择你的分支
4. 填写 PR 描述模板
5. 等待 CI 运行和代码审查

## 代码标准

### 命名约定
- **函数**: `snake_case`
- **变量**: `snake_case`
- **类型**: `snake_type_t`
- **常量**: `SNAKE_CASE`
- **文件**: `snake_case.c`, `snake_case.h`

### 代码风格
- **缩进**: 4 个空格（非制表符）
- **行宽**: 最大 100 字符
- **大括号**: K&R 风格
- **注释**: 全英文，无中文字符

### 示例
```c
// 函数声明
boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim,
                                  boat_dtype_t dtype, boat_device_t device);

// 结构体定义
struct boat_tensor_t {
    int64_t* shape;
    size_t ndim;
    boat_dtype_t dtype;
};

// 常量定义
#define BOAT_MAX_DIMENSIONS 8
```

## 质量要求

### 1. 代码正确性
- 通过所有现有测试
- 添加新功能的测试覆盖率
- 处理边界条件和错误情况

### 2. 内存安全
- 无内存泄漏（使用 Valgrind 检查）
- 无悬空指针
- 正确的引用计数管理

### 3. 性能考虑
- 避免不必要的内存分配
- 使用高效算法
- 考虑缓存友好性

### 4. 可维护性
- 清晰的代码结构
- 有意义的变量名
- 适当的注释（解释为什么，而不是做什么）

## 测试要求

### 单元测试
- 每个新功能应有对应的单元测试
- 测试应覆盖正常情况和错误情况
- 测试文件命名: `test_模块名.c`

### 集成测试
- 验证模块间的交互
- 测试端到端功能
- 确保向后兼容性

### 测试结构
```c
#include <boat/test.h>

TEST(test_function_name) {
    // 测试代码
    ASSERT(condition, "错误信息");
    ASSERT_EQ(expected, actual);
    ASSERT_NEAR(float_expected, float_actual, epsilon);
}

int main() {
    RUN_TEST(test_function_name);
    return 0;
}
```

## 文档要求

### 代码文档
- 公共 API 必须有文档注释
- 复杂算法应有解释性注释
- 头文件应描述模块用途

### API 文档示例
```c
/**
 * 创建新张量
 *
 * @param shape 张量形状数组，长度为 ndim
 * @param ndim 张量维度数（0 表示标量）
 * @param dtype 数据类型
 * @param device 设备类型（CPU/GPU）
 * @return 新张量指针，失败返回 NULL
 */
BOAT_API boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim,
                                           boat_dtype_t dtype, boat_device_t device);
```

### 用户文档
- 更新相关文档（如需要）
- 添加使用示例
- 更新 README（如影响安装或使用）

## 审查流程

### PR 审查标准
1. **功能正确性**: 实现是否符合需求
2. **代码质量**: 是否符合代码标准
3. **测试覆盖**: 是否有充分测试
4. **文档完整**: 是否更新相关文档
5. **性能影响**: 是否影响现有性能

### 审查反馈
- 建设性批评
- 具体改进建议
- 解释审查决定的原因

### 常见审查意见
- **需要更多测试**: 添加测试用例
- **代码风格问题**: 遵循风格指南
- **缺少文档**: 添加 API 文档
- **性能问题**: 优化算法或内存使用

## 特殊贡献类型

### 错误修复
1. 创建最小复现用例
2. 定位根本原因
3. 提供修复方案
4. 添加回归测试

### 性能优化
1. 提供性能基准
2. 证明优化效果
3. 确保不破坏功能
4. 更新相关文档

### 文档改进
1. 确保信息准确
2. 保持风格一致
3. 添加实用示例
4. 检查链接有效性

## 工具支持

### 开发工具
- **编辑器配置**: `.editorconfig` 文件
- **代码格式化**: `clang-format` 配置
- **静态分析**: `cppcheck` 集成
- **构建系统**: CMake

### 本地检查脚本
```bash
# 运行代码格式化检查
./scripts/check_format.sh

# 运行静态分析
./scripts/check_static.sh

# 运行完整测试套件
./scripts/run_tests.sh
```

## 社区准则

### 行为准则
1. **尊重**: 尊重所有社区成员
2. **包容**: 欢迎不同背景的贡献者
3. **协作**: 积极协作，共同解决问题
4. **专业**: 保持专业和技术性讨论

### 沟通渠道
- **GitHub Issues**: 问题报告和功能请求
- **GitHub Discussions**: 技术讨论和问答
- **Pull Requests**: 代码贡献和审查

## 快速参考

### 常用命令
```bash
# 设置开发环境
git clone --recursive https://github.com/your-username/boat.git
cd boat
mkdir build && cd build
cmake .. -DBOAT_WITH_TESTS=ON -DCMAKE_BUILD_TYPE=Debug

# 日常开发循环
make                    # 构建
ctest -V               # 运行测试
cppcheck src          # 静态检查

# 提交更改
git add .
git commit -m "feat: 添加新功能"
git push origin feature/xxx
```

### 资源链接
- [开发者入门指南](developer_getting_started.md)
- [CI/CD 指南](ci_cd_guide.md)
- [性能优化指南](performance_optimization_guide.md)
- [Const 使用指南](const_usage_guide.md)
- [代码风格详细说明](CLAUDE.md)

## 问题解决

### 常见问题
**Q: 我的 PR 被拒绝，我该怎么办？**
A: 仔细阅读审查意见，修改代码后重新提交。如有疑问，礼貌地请求澄清。

**Q: 如何添加新的依赖？**
A: 在 CMakeLists.txt 中添加依赖，并更新文档。重大依赖变更需要讨论。

**Q: 测试在我的机器上通过，但在 CI 中失败？**
A: 检查跨平台兼容性问题，确保测试不依赖于特定环境。

**Q: 我的贡献何时会被合并？**
A: 这取决于 PR 的复杂性、审查进度和项目优先级。通常需要 1-2 周。

---

*本指南最后更新: 2026-03-01*
*欢迎贡献改进建议！*