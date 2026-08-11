# Boat Framework Const Correctness Usage Guide

## Overview

This guide defines the correct usage strategy for the `const` keyword in the Boat deep learning framework. Proper `const` usage improves code safety, readability, and the compiler's ability to optimize.

## Core Principles

### 1. Tiered Const Strategy
- **Read-only functions (Reader/Getter)**: Mark parameters and return values with const
- **Mutating functions (Writer/Setter)**: Use non-const parameters
- **Ownership-transfer functions**: Use non-const parameters (ownership must be modified)

### 2. Function Classification

#### 2.1 Forward Propagation Functions (Forward Pass)
```c
// Correct: forward pass does not modify layer state, use const
boat_tensor_t* boat_attention_forward(const boat_attention_t* attention,
                                     const boat_tensor_t* query,
                                     const boat_tensor_t* key,
                                     const boat_tensor_t* value,
                                     const boat_tensor_t* attention_mask);
```

#### 2.2 Backward Propagation Functions (Backward Pass)
```c
// Correct: backward pass needs gradient storage, use non-const
bool boat_attention_backward(boat_attention_t* attention,
                            const boat_tensor_t* grad_output,
                            boat_tensor_t** grad_query,
                            boat_tensor_t** grad_key,
                            boat_tensor_t** grad_value);
```

#### 2.3 Parameter Update Functions
```c
// Correct: update internal parameters, use non-const
void boat_attention_update(boat_attention_t* attention, float learning_rate);
```

#### 2.4 Memory Management Functions
```c
// Correct: freeing memory requires modifying ownership, use non-const
void boat_attention_free(boat_attention_t* attention);
```

#### 2.5 Accessor Functions (Accessors)
```c
// Correct: read-only access, use const
boat_tensor_t* boat_attention_get_weight_q(const boat_attention_t* attention);
```

## API Design Guidelines

### 1. Parameter Passing Rules

| Parameter Type | Const Qualifier | Example | Reason |
|---------|-----------|------|------|
| Input parameter (read-only) | `const type*` | `const boat_tensor_t* input` | The function does not modify the parameter |
| Output parameter (writable) | `type*` | `boat_tensor_t** grad_output` | The function needs to write the result |
| Input/output parameter | `type*` | `boat_attention_t* attention` | The function both reads and modifies the parameter |
| Scalar parameter | Pass by value | `float learning_rate` | Small types are passed by value |

### 2. Return Value Rules

| Return Value Type | Const Qualifier | Example | Reason |
|-----------|-----------|------|------|
| Newly allocated object | `type*` | `boat_tensor_t*` | The caller gains ownership |
| Reference to internal object | `const type*` | `const boat_tensor_t*` | Read-only access; the caller does not gain ownership |
| Boolean/status | Pass by value | `bool` | Small types are passed by value |

### 3. Struct Field Rules

```c
typedef struct boat_layer_t {
    void* data;                    // internal data, mutable
    const boat_layer_ops_t* ops;   // op table, read-only (like vtable)
} boat_layer_t;
```

## Common Patterns

### 1. Create-Use-Destroy Pattern
```c
// create: returns new object
boat_attention_t* attn = boat_attention_create(&config);

// use: forward (const), backward (non-const)
boat_tensor_t* output = boat_attention_forward(attn, query, key, value, NULL);
bool success = boat_attention_backward(attn, grad_output, &grad_q, &grad_k, &grad_v);

// destroy: non-const required
boat_attention_free(attn);
```

### 2. Getter/Setter Pattern
```c
// Getter: const param, returns const or non-const pointer (based on ownership)
boat_tensor_t* weight = boat_attention_get_weight_q(attn);  // returns internal reference

// Setter: non-const param
void boat_attention_set_dropout(boat_attention_t* attn, float prob);
```

## Compiler Compatibility

### 1. MSVC-Specific Issues
MSVC performs stricter const correctness checks, especially:
- lvalue specifies const object error (C2166)
- function signatures must be consistent between the actual definition and the declaration

### 2. Cross-Compiler Strategy
- All public API headers must explicitly specify const qualifiers
- Function definitions in implementation files must match their declarations exactly
- Avoid using `const_cast` in implementations to bypass checks

## Error Handling

### 1. Common Errors
```c
// Wrong: calling mutating function with const param
const boat_attention_t* attn = boat_attention_create(&config);
boat_attention_set_dropout(attn, 0.5f);  // compile error: attn is const

// Correct: use non-const pointer
boat_attention_t* attn = boat_attention_create(&config);
boat_attention_set_dropout(attn, 0.5f);  // correct
```

### 2. Debugging Suggestions
- Use the compiler's `-Wcast-qual` option (GCC/Clang)
- Periodically run cppcheck to verify const correctness
- Pay special attention to const usage during code review

## Migration Guide

### 1. From Non-const to const
1. Identify read-only functions and add const qualifiers
2. Update caller code to pass const pointers
3. Resolve compilation errors, distinguishing cases that truly need modification

### 2. Backward Compatibility
- Avoid abruptly changing the const qualifiers of existing APIs
- If changes are needed, provide a transition period and documentation
- Consider providing compatibility wrapper functions

## Example

### Complete Example: Attention Layer
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

## Summary

The Boat framework adopts a tiered const strategy:
1. **Use const for read-only operations**: improves safety and compiler optimization
2. **Use non-const for mutating operations**: clearly expresses intent
3. **Use non-const for ownership transfer**: avoids confusion

Following these guidelines produces safer, clearer, and more efficient code.

---
**Document version**: 1.0
**Last updated**: 2026-03-01
**Applicable version**: Boat framework v0.1.0+
**Maintainer**: Engineer Xiao
