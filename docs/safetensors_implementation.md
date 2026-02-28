# Safetensors 格式支持实现总结

## 概述

为 Boat C 语言深度学习框架实现了 Hugging Face safetensors 模型格式的基本支持。safetensors 是 Hugging Face 推荐的模型权重存储格式，具有简单、安全、高效的特点。

## 已完成的实现

### 1. safetensors 解析器核心功能
- **头部解析**：实现了 `safetensors` 格式的 8 字节头部长度解析
- **数据类型映射**：添加了 `safetensors_dtype_t` 枚举和 `boat_dtype_t` 转换函数
- **JSON 解析**：集成了 cJSON 库用于解析 safetensors 的 JSON 头部
- **张量加载**：实现了从二进制数据加载张量到 `boat_tensor_t` 的功能

### 2. cJSON 集成配置
- **CMake 选项**：添加了 `BOAT_CJSON_PATH` 选项支持自定义 cJSON 路径
- **条件编译**：通过 `BOAT_USE_CJSON` 宏控制 cJSON 功能
- **路径支持**：支持从自定义路径（如 `D:\github\cJSON`）加载 cJSON

### 3. 关键函数实现
- `parse_safetensors_header()`：解析 safetensors 文件头部
- `load_safetensors()`：加载 safetensors 权重文件
- `parse_config()`：解析 Hugging Face 的 config.json
- `create_layer_from_config()`：张量名称到 Boat 层的映射框架

### 4. 构建系统更新
```cmake
# 编译时启用 Hugging Face 支持
cmake -DBOAT_WITH_HUGGINGFACE=ON -DBOAT_CJSON_PATH="D:/github/cJSON" ..
make
```

## 使用示例

### 从内存加载模型
```c
#include <boat/format/huggingface.h>

// 1. 读取 config.json 到字符串
const char* config_json = "{ \"model_type\": \"bert\", ... }";

// 2. 读取 model.safetensors 到内存
void* weights_data = ...;
size_t weights_size = ...;

// 3. 加载模型
boat_model_t* model = boat_huggingface_load_from_memory(config_json, weights_data, weights_size);
```

### safetensors 格式支持
- **文件结构**：8 字节(头部长度) + JSON 头部 + 二进制张量数据
- **数据类型**：支持 F32、F16、I32、I64、U8、BOOL 等常见类型
- **张量元数据**：解析 dtype、shape、data_offsets 字段

## 需要完善的下一步

### 1. 层映射实现
`create_layer_from_config()` 函数需要根据具体的 Hugging Face 模型架构实现：
- **BERT 模型**：embeddings、attention、dense、layer_norm 层映射
- **GPT-2 模型**：causal attention、feed-forward 网络映射
- **CNN 模型**：卷积层、池化层映射

### 2. 文件系统加载
`boat_huggingface_load()` 函数需要实现文件读取：
- 读取 `config.json` 文件
- 检测 `model.safetensors` 或 `pytorch_model.bin`
- 调用内存加载接口

### 3. 模型架构支持
当前框架支持通用的 safetensors 解析，但需要针对具体模型类型：
- **BERT-base**：12 层 Transformer，768 隐藏层
- **GPT-2**：解码器架构，因果注意力
- **RoBERTa**：BERT 变体，无 NSP 任务

## 测试建议

### 1. 编译测试
```bash
cd build
cmake -DBOAT_WITH_HUGGINGFACE=ON -DBOAT_CJSON_PATH="D:/github/cJSON" ..
make
```

### 2. 使用 MNIST CNN 模型测试
已有的 `safetensors` 文件可用于测试：
```c
// 加载 MNIST CNN 模型的 safetensors 文件
boat_model_t* model = boat_huggingface_load_from_memory(config_json, safetensors_data, data_size);
```

### 3. 验证张量加载
当前实现会打印加载的张量信息：
```
Loaded tensor 'conv1.weight' with shape [32, 1, 3, 3] dtype=F32
Loaded tensor 'conv1.bias' with shape [32] dtype=F32
```

## 技术细节

### safetensors 格式
```python
# 文件结构
[8字节: JSON长度][JSON头部][张量数据...]

# JSON头部示例
{
  "__header__": {"format": "pt"},
  "conv1.weight": {
    "dtype": "F32",
    "shape": [32, 1, 3, 3],
    "data_offsets": [0, 1152]
  }
}
```

### 数据类型映射
| safetensors | boat_dtype_t | 描述 |
|-------------|--------------|------|
| F32 | BOAT_DTYPE_FLOAT32 | 32位浮点数 |
| F16 | BOAT_DTYPE_FLOAT16 | 16位浮点数 |
| I32 | BOAT_DTYPE_INT32 | 32位整数 |
| U8 | BOAT_DTYPE_UINT8 | 8位无符号整数 |

### 文件格式验证
- **头部长度**：前 8 字节（小端序）表示 JSON 头部长度
- **JSON 结构**：必须包含 `__header__` 和各个张量的元数据
- **数据对齐**：张量数据按 8 字节对齐存储
- **安全检查**：验证偏移范围和数据边界

## 实现状态

### ✅ 已完成
- safetensors 头部解析
- JSON 元数据提取
- 张量数据加载
- cJSON 集成配置

### ⏳ 待实现
- 完整的层映射系统
- 文件系统加载接口
- 特定模型架构支持
- 错误处理和验证

### 🔄 进行中
- BERT 模型层映射
- 配置文件解析优化

## 性能考虑
- **内存效率**：直接映射文件数据，减少内存拷贝
- **解析速度**：使用 cJSON 快速解析 JSON 头部
- **张量创建**：复用现有 `boat_tensor_t` 创建接口

## 扩展性设计
- **模块化架构**：解析器与层映射分离
- **可插拔后端**：支持 cJSON 和简单解析器两种模式
- **渐进实现**：从简单模型开始，逐步支持复杂架构

---

*文档创建时间：2026-02-22*
*最后更新：2026-02-22*