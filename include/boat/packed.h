// packed.h - Packed data type utilities for deep learning framework
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_PACKED_H
#define BOAT_PACKED_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packed format utilities for low-bit types

// BITS1 (1-bit) packing/unpacking
// Each byte stores 8 binary values (0 or 1)
void boat_pack_bits1(const bool* src, uint8_t* dst, size_t n);
void boat_unpack_bits1(const uint8_t* src, bool* dst, size_t n);

// BITS2 (2-bit) packing/unpacking
// Each byte stores 4 values (0-3)
void boat_pack_bits2(const uint8_t* src, uint8_t* dst, size_t n);
void boat_unpack_bits2(const uint8_t* src, uint8_t* dst, size_t n);

// FLOAT4 (4-bit) custom floating point format
// Each byte stores 2 values
// Format: 1 sign bit, 3 exponent bits, 0 mantissa bits (custom)
typedef struct {
    uint8_t sign : 1;
    uint8_t exponent : 3;
    // No mantissa bits for 4-bit float
} boat_float4_t;

boat_float4_t boat_float4_from_float(float f);
float boat_float4_to_float(boat_float4_t f4);

void boat_pack_float4(const float* src, uint8_t* dst, size_t n);
void boat_unpack_float4(const uint8_t* src, float* dst, size_t n);

// FLOAT8 (8-bit) custom floating point format
// Format: 1 sign bit, 4 exponent bits, 3 mantissa bits (custom)
typedef struct {
    uint8_t sign : 1;
    uint8_t exponent : 4;
    uint8_t mantissa : 3;
} boat_float8_t;

boat_float8_t boat_float8_from_float(float f);
float boat_float8_to_float(boat_float8_t f8);

void boat_pack_float8(const float* src, uint8_t* dst, size_t n);
void boat_unpack_float8(const uint8_t* src, float* dst, size_t n);

// Element-wise operations for packed types
// These functions operate directly on packed data
void boat_add_bits1(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes);
void boat_add_bits2(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes);
void boat_add_float4(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes);
void boat_add_float8(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes);

// Memory access optimizations for packed types
size_t boat_packed_element_offset(boat_dtype_t dtype, size_t index);
uint8_t boat_packed_read_element(const uint8_t* data, boat_dtype_t dtype, size_t index);
void boat_packed_write_element(uint8_t* data, boat_dtype_t dtype, size_t index, uint8_t value);

#ifdef __cplusplus
}
#endif

#endif // BOAT_PACKED_H