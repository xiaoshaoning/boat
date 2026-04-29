# Boat 框架 Const 正确性使用指南

## 概述

本指南定义了 Boat 深度学习框架中 const 关键字的正确使用策略。正确的 const 使用有助于提高代码安全性、可读性和编译器优化能力。

## 核心原则

### 1. 分层 Const 策略
- **只读函数 (Reader/Getter)**: 使用 const 修饰参数和返回值
- **修改函数 (Writer/Setter)**: 使用非 const 参数
- **所有权转移函数**: 使用非 const 参数（需要修改所有权）

### 2. 函数分类

#### 2.1 前向传播函数 (Forward Pass)
```c
// Correct: forward pass does not modify layer state, use const
boat_tensor_t* boat_attention_forward(const boat_attention_t* attention,
                                     const boat_tensor_t* query,
                                     const boat_tensor_t* key,
                                     const boat_tensor_t* value,
                                     const boat_tensor_t* attention_mask);
```

#### 2.2 反向传播函数 (Backward Pass)
```c
// Correct: backward pass needs gradient storage, use non-const
bool boat_attention_backward(boat_attention_t* attention,
                            const boat_tensor_t* grad_output,
                            boat_tensor_t** grad_query,
                            boat_tensor_t** grad_key,
                            boat_tensor_t** grad_value);
```

#### 2.3 参数更新函数
```c
// Correct: update internal parameters, use non-const
void boat_attention_update(boat_attention_t* attention, float learning_rate);
```

#### 2.4 内存管理函数
```c
// Correct: freeing memory requires modifying ownership, use non-const
void boat_attention_free(boat_attention_t* attention);
```

#### 2.5 访问器函数 (Accessors)
```c
// Correct: read-only access, use const
boat_tensor_t* boat_attention_get_weight_q(const boat_attention_t* attention);
```

## API 设计规范

### 1. 参数传递规则

| 参数类型 | Const 修饰 | 示例 | 理由 |
|---------|-----------|------|------|
| 输入参数 (只读) | `const type*` | `const boat_tensor_t* input` | 函数不修改参数内容 |
| 输出参数 (可写) | `type*` | `boat_tensor_t** grad_output` | 函数需要写入结果 |
| 输入输出参数 | `type*` | `boat_attention_t* attention` | 函数既读取又修改参数 |
| 标量参数 | 按值传递 | `float learning_rate` | 小类型按值传递 |

### 2. 返回值规则

| 返回值类型 | Const 修饰 | 示例 | 理由 |
|-----------|-----------|------|------|
| 新分配对象 | `type*` | `boat_tensor_t*` | 调用者获得所有权 |
| 内部对象引用 | `const type*` | `const boat_tensor_t*` | 只读访问，调用者不获得所有权 |
| 布尔/状态 | 按值传递 | `bool` | 小类型按值传递 |

### 3. 结构体字段规则

```c
typedef struct boat_layer_t {
    void* data;                    // internal data, mutable
    const boat_layer_ops_t* ops;   // op table, read-only (like vtable)
} boat_layer_t;
```

## 常见模式

### 1. 创建-使用-销毁模式
```c
// create: returns new object
boat_attention_t* attn = boat_attention_create(&config);

// use: forward (const), backward (non-const)
boat_tensor_t* output = boat_attention_forward(attn, query, key, value, NULL);
bool success = boat_attention_backward(attn, grad_output, &grad_q, &grad_k, &grad_v);

// destroy: non-const required
boat_attention_free(attn);
```

### 2. Getter/Setter 模式
```c
// Getter: const param, returns const or non-const pointer (based on ownership)
boat_tensor_t* weight = boat_attention_get_weight_q(attn);  // returns internal reference

// Setter: non-const param
void boat_attention_set_dropout(boat_attention_t* attn, float prob);
```

## 编译器兼容性

### 1. MSVC 特定问题
MSVC 对 const 正确性检查较为严格，特别是：
- 左值指定 const 对象错误 (C2166)
- 需要确保函数签名在实际定义和声明中一致

### 2. 跨编译器策略
- 所有公共 API 头文件必须明确定义 const 修饰
- 实现文件中的函数定义必须与声明完全匹配
- 避免在实现中使用 `const_cast` 绕过检查

## 错误处理

### 1. 常见错误
```c
// Wrong: calling mutating function with const param
const boat_attention_t* attn = boat_attention_create(&config);
boat_attention_set_dropout(attn, 0.5f);  // compile error: attn is const

// Correct: use non-const pointer
boat_attention_t* attn = boat_attention_create(&config);
boat_attention_set_dropout(attn, 0.5f);  // correct
```

### 2. 调试建议
- 使用编译器的 `-Wcast-qual` 选项（GCC/Clang）
- 定期运行 cppcheck 检查 const 正确性
- 在代码审查中特别关注 const 使用

## 迁移指南

### 1. 从非 const 到 const
1. 识别只读函数，添加 const 修饰符
2. 更新调用方代码，确保传递 const 指针
3. 处理编译错误，区分真正需要修改的情况

### 2. 向后兼容性
- 避免突然改变现有 API 的 const 修饰
- 如果需要更改，提供过渡期和文档说明
- 考虑提供兼容性包装函数

## 示例

### 完整示例：注意力层
```c
// create (non-const return)
boat_attention_t* attn = boat_attention_create(&config);

// forward (const param)
boat_tensor_t* output = boat_attention_forward(attn, query, key, value, NULL);

// access weights (const param, returns non-const pointer)
boat_tensor_t* weight_q = boat_attention_get_weight_q(attn);

// modify config (non-const param)
boat_attention_set_dropout(attn, 0.1f);

// backward (non-const param)
boat_tensor_t* grad_q, *grad_k, *grad_v;
bool success = boat_attention_backward(attn, grad_output, &grad_q, &grad_k, &grad_v);

// update parameters (non-const param)
boat_attention_update(attn, 0.001f);

// destroy (non-const param)
boat_attention_free(attn);
```

## 总结

Boat 框架采用分层 const 策略：
1. **只读操作使用 const**：提高安全性和编译器优化
2. **修改操作使用非 const**：明确表达意图
3. **所有权转移使用非 const**：避免混淆

遵循这些规范将产生更安全、更清晰、更高效的代码。

---
**文档版本**: 1.0
**更新日期**: 2026-03-01
**适用版本**: Boat 框架 v0.1.0+
**维护者**: 萧工