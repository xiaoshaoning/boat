# Boat 性能优化指南

## 概述

本指南提供 Boat 深度学习框架的性能优化策略、最佳实践和调优技巧，帮助开发者编写高效代码。

## 性能原则

### 核心原则
1. **测量优先**: 在优化前测量性能瓶颈
2. **渐进优化**: 从算法优化开始，再到微优化
3. **权衡考虑**: 平衡性能、可读性和可维护性
4. **平台感知**: 考虑 CPU/GPU 架构差异

### 性能层级
1. **算法层面**: 选择高效算法 (O(n) vs O(n²))
2. **内存层面**: 优化内存访问模式
3. **指令层面**: 减少指令数量，利用 SIMD
4. **系统层面**: 并行化、缓存优化

## 性能测量工具

### 时间测量
```c
#include <time.h>

clock_t start = clock();
// 需要测量的代码
clock_t end = clock();
double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
printf("Time: %f seconds\n", elapsed);
```

### 内存测量
```c
#include <stdlib.h>
#include <stdio.h>

size_t start_memory = get_current_memory_usage();
// 内存操作
size_t end_memory = get_current_memory_usage();
printf("Memory delta: %zu bytes\n", end_memory - start_memory);
```

### 性能分析工具
- **Linux**: `perf`, `valgrind --tool=callgrind`, `gprof`
- **macOS**: Instruments, `sample`
- **Windows**: Visual Studio Profiler, Windows Performance Toolkit
- **跨平台**: `google/benchmark` 库

## 张量操作优化

### 内存布局
- **连续内存**: 确保张量数据在内存中连续存储
- **缓存友好**: 优化数据访问模式，提高缓存命中率
- **对齐**: 确保内存对齐，支持 SIMD 指令

### 示例优化
```c
// 低效: 每次计算索引
for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
        data[i * stride_i + j * stride_j] = ...;
    }
}

// 高效: 预计算指针
for (size_t i = 0; i < n; i++) {
    float* row = data + i * stride_i;
    for (size_t j = 0; j < m; j++) {
        row[j * stride_j] = ...;
    }
}
```

### 批量操作
- 使用批量处理减少函数调用开销
- 合并小操作为大操作
- 利用向量化指令

## 内存管理优化

### 分配策略
1. **池化分配**: 复用内存块，减少 malloc/free 调用
2. **预分配**: 预先分配足够内存，避免频繁重分配
3. **对齐分配**: 使用对齐的内存分配，提高 SIMD 性能

### 示例: 内存池
```c
typedef struct {
    void** blocks;
    size_t capacity;
    size_t size;
} memory_pool_t;

void* pool_alloc(memory_pool_t* pool, size_t size) {
    if (pool->size >= pool->capacity) {
        // 扩展池
    }
    return pool->blocks[pool->size++];
}

void pool_reset(memory_pool_t* pool) {
    pool->size = 0;
}
```

### 减少内存碎片
- 使用固定大小分配器
- 避免频繁的小内存分配
- 定期整理内存

## 计算优化

### 循环优化
```c
// 循环展开
for (size_t i = 0; i < n; i += 4) {
    data[i] = ...;
    data[i+1] = ...;
    data[i+2] = ...;
    data[i+3] = ...;
}

// 减少循环内计算
size_t stride = calculate_stride();
for (size_t i = 0; i < n; i++) {
    // 避免在循环内重复计算
    size_t offset = i * stride;  // 好: 在循环外计算
    // vs
    size_t offset = calculate_offset(i);  // 差: 每次调用函数
}
```

### 数学优化
- 使用查表法替代复杂计算
- 近似计算，在可接受误差范围内
- 利用数学恒等式简化表达式

### SIMD 优化
```c
#ifdef __AVX2__
#include <immintrin.h>

void vector_add(float* a, float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(&c[i], vc);
    }
}
#endif
```

## 并行化优化

### OpenMP 集成
```c
#include <omp.h>

#pragma omp parallel for
for (size_t i = 0; i < n; i++) {
    // 并行化循环
}
```

### 线程池
- 创建线程池，避免频繁创建销毁线程
- 任务队列管理并行任务
- 负载均衡，避免线程饥饿

### 数据并行 vs 任务并行
- **数据并行**: 相同操作应用于不同数据 (适合 SIMD/GPU)
- **任务并行**: 不同操作并行执行 (适合多核 CPU)

## GPU 优化 (未来支持)

### 内存传输优化
- 最小化主机-设备数据传输
- 使用 pinned memory 加速传输
- 异步传输与计算重叠

### 内核优化
- 优化线程网格配置
- 使用共享内存减少全局内存访问
- 避免线程发散 (warp divergence)

## 编译器优化

### 编译标志
```cmake
# Release 模式优化
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# 链接时优化
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
```

### 内联优化
```c
// 使用 static inline 提示编译器
static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
```

### 分支预测
```c
// 提示编译器分支概率
if (likely(condition)) {  // 很可能为真
    // 快速路径
} else {
    // 慢速路径
}
```

## 框架特定优化

### 计算图优化
1. **操作融合**: 合并多个操作为一个内核
2. **常量折叠**: 预先计算常量表达式
3. **死代码消除**: 移除无用的计算节点
4. **公共子表达式消除**: 复用重复计算结果

### 自动微分优化
- 反向传播内存复用
- 梯度计算流水线化
- 检查点技术，平衡内存与计算

### 模型加载优化
- 懒加载模型参数
- 并行加载大型模型
- 使用内存映射文件

## 性能测试套件

### 基准测试框架
```c
#include <boat/benchmark.h>

BENCHMARK("矩阵乘法", [](BenchmarkState& state) {
    size_t n = state.range(0);
    Matrix a = create_random_matrix(n, n);
    Matrix b = create_random_matrix(n, n);

    for (auto _ : state) {
        Matrix c = matrix_multiply(a, b);
        benchmark::DoNotOptimize(c);
    }
})->Range(64, 1024)->Unit(benchmark::kMillisecond);
```

### 性能回归测试
- 每次提交运行性能基准
- 检测性能回归
- 设置性能阈值

### 监控与报警
- 记录历史性能数据
- 设置性能退化报警
- 可视化性能趋势

## 优化案例研究

### 案例 1: 矩阵乘法优化
**问题**: 朴素实现 O(n³) 复杂度
**优化**:
1. 使用分块算法提高缓存命中率
2. 使用 SIMD 指令向量化
3. 使用多线程并行化
**结果**: 速度提升 20 倍

### 案例 2: 激活函数优化
**问题**: `expf()` 函数调用开销大
**优化**:
1. 使用近似公式: `sigmoid(x) ≈ 0.5 * (x / (1 + |x|)) + 0.5`
2. 使用查表法预处理
3. 向量化计算
**结果**: 速度提升 5 倍，误差 < 0.1%

### 案例 3: 内存分配优化
**问题**: 训练循环中频繁分配释放内存
**优化**:
1. 实现张量内存池
2. 复用前向传播内存用于反向传播
3. 预分配最大所需内存
**结果**: 内存分配开销减少 90%

## 最佳实践清单

### 开发时
- [ ] 编写可读代码，然后优化
- [ ] 添加性能测试
- [ ] 使用性能分析工具定位瓶颈
- [ ] 考虑算法复杂度

### 优化时
- [ ] 一次优化一个瓶颈
- [ ] 验证优化后功能正确性
- [ ] 测量优化前后性能
- [ ] 考虑不同硬件平台

### 维护时
- [ ] 定期运行性能测试
- [ ] 监控性能回归
- [ ] 更新优化指南
- [ ] 分享优化经验

## 工具与资源

### 分析工具
- **Profiler**: `perf`, `gprof`, `VTune`
- **Memory**: `valgrind`, `AddressSanitizer`
- **Cache**: `cachegrind`, `perf c2c`

### 优化库
- **SIMD**: Intel IPP, ARM NEON 内在函数
- **并行**: OpenMP, Intel TBB
- **数学**: Intel MKL, OpenBLAS

### 学习资源
- [Intel 优化手册](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [ARM 优化指南](https://developer.arm.com/documentation)
- [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## 性能调优工作流

1. **定位瓶颈**: 使用分析工具找到热点
2. **分析原因**: 理解性能限制因素（CPU、内存、I/O）
3. **设计优化**: 选择合适的优化策略
4. **实施优化**: 编写优化代码
5. **验证结果**: 测试功能和性能
6. **集成监控**: 添加性能监控和报警

## 注意事项

### 避免过早优化
- 首先确保代码正确
- 优化显著的瓶颈，而非微观优化
- 保持代码可读性和可维护性

### 平台兼容性
- 为不同平台提供优化实现
- 运行时检测硬件特性
- 提供回退到通用实现

### 测试覆盖
- 优化后运行完整测试套件
- 验证数值精度可接受
- 确保边缘情况正确处理

## 贡献优化

欢迎贡献性能优化！请遵循：
1. 提供性能测量数据（优化前后对比）
2. 确保不破坏现有功能
3. 添加适当的测试
4. 更新相关文档

---

*本指南最后更新: 2026-03-01*
*性能优化是持续过程，欢迎分享经验和改进建议！*