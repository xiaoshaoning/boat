# Hugging Face Transformer 模型格式支持实现总结

## 概述

为 C 语言深度学习框架 Boat 添加了对 Hugging Face Transformer 模型格式的基本支持。当前实现提供了 API 框架和构建系统集成，主要功能实现待完成。

## 已添加的文件

### 1. 头文件 `include/boat/format/huggingface.h`
- 定义了加载 Hugging Face 模型的 API 接口
- 支持从目录加载和从内存加载两种方式

### 2. 源文件 `src/format/huggingface.c`
- 实现了 API 的骨架结构
- 包含了 Hugging Face 配置解析的数据结构
- 预留了权重加载函数接口

### 3. CMake 配置更新
- 添加了 `BOAT_WITH_HUGGINGFACE` 编译选项
- 添加了对 cJSON 库的依赖（用于 JSON 解析）
- 自动将 huggingface.c 添加到构建系统

## 使用方式

### 编译时启用 Hugging Face 支持：
```bash
cmake -DBOAT_WITH_HUGGINGFACE=ON ..
make
```

### API 用法示例：
```c
#include <boat/format/huggingface.h>

// 从目录加载 Hugging Face 模型
boat_model_t* model = boat_huggingface_load("path/to/model_dir");
// model_dir 应包含 config.json 和权重文件（pytorch_model.bin 或 model.safetensors）
```

## 需要实现的下一步

当前的实现只是骨架，需要完成以下关键功能：

### 1. 集成 cJSON 库
- 安装 cJSON：`apt-get install libcjson-dev` 或从源码编译
- 在 huggingface.c 中取消注释 `#include "cJSON.h"`

### 2. 解析 config.json
- 实现 `parse_config()` 函数，解析 Hugging Face 模型配置
- 关键字段：`model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads` 等
- 根据 model_type 确定模型架构（BERT、GPT-2、RoBERTa 等）

### 3. 权重文件加载

**方案 A：支持 safetensors 格式（推荐）**
- 格式简单安全，纯二进制 + JSON 头
- 实现 `load_safetensors()` 函数
- 解析格式：https://huggingface.co/docs/safetensors

**方案 B：支持 PyTorch .bin 格式**
- 复杂，需要解析 Pickle 格式
- 可考虑使用 LibTorch C++ API 辅助解析

### 4. 层映射系统
- 实现 `create_layer_from_config()` 函数
- 将 Hugging Face 层映射到 Boat 层：
  - `Linear` → `boat_dense_layer_t`
  - `LayerNorm` → `boat_norm_layer_t`
  - `Embedding` → 自定义嵌入层
  - `Attention` → `boat_attention_layer_t`

### 5. 权重分配
- 根据权重名称（如 `bert.encoder.layer.0.attention.self.query.weight`）找到对应层
- 使用层的参数设置函数（如 `boat_dense_layer_set_weight()`）分配权重

## 建议的实现顺序

1. **先支持 safetensors 格式**（比 PyTorch .bin 简单）
2. **先支持一种模型类型**（如 BERT-base）
3. **逐步扩展**到其他模型架构

## 依赖管理

cJSON 是必须的依赖。你可以：

- 作为系统包安装
- 作为 git 子模块添加到项目中
- 直接包含源码到 `external/` 目录

需要修改 CMakeLists.txt 确保正确找到 cJSON：
```cmake
find_package(cJSON REQUIRED)
target_link_libraries(boat PUBLIC cJSON::cJSON)
```

## 当前实现状态

- ✅ API 接口定义完成
- ✅ 构建系统集成完成
- ⏳ cJSON 集成待完成
- ⏳ config.json 解析待实现
- ⏳ 权重文件加载待实现
- ⏳ 层映射系统待实现

## 下一步行动计划

1. **第 1 步**：集成 cJSON 库并测试 JSON 解析
2. **第 2 步**：实现 config.json 解析器
3. **第 3 步**：实现 safetensors 格式解析器
4. **第 4 步**：实现 BERT 模型的层映射和权重加载
5. **第 5 步**：测试加载简单的 BERT 模型进行推理
6. **第 6 步**：扩展支持其他模型类型

---

*文档创建时间：2026-02-22*
*最后更新：2026-02-22*