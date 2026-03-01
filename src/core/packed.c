// packed.c - Packed data type utilities implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/packed.h>
#include <boat/memory.h>
#include <string.h>
#include <math.h>

// BITS1 packing/unpacking
void boat_pack_bits1(const bool* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (size_t byte_idx = 0; byte_idx < (n + 7) / 8; byte_idx++) {
        uint8_t byte = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (i < n && src[i]) {
                byte |= (1 << bit);
            }
            i++;
        }
        dst[byte_idx] = byte;
    }
}

void boat_unpack_bits1(const uint8_t* src, bool* dst, size_t n) {
    size_t i = 0;
    for (size_t byte_idx = 0; byte_idx < (n + 7) / 8; byte_idx++) {
        uint8_t byte = src[byte_idx];
        for (int bit = 0; bit < 8; bit++) {
            if (i < n) {
                dst[i] = (byte >> bit) & 1;
                i++;
            } else {
                break;
            }
        }
    }
}

// BITS2 packing/unpacking
void boat_pack_bits2(const uint8_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (size_t byte_idx = 0; byte_idx < (n + 3) / 4; byte_idx++) {
        uint8_t byte = 0;
        for (int nibble = 0; nibble < 4; nibble++) {
            if (i < n) {
                uint8_t val = src[i] & 0x03; // Ensure 2-bit value
                byte |= (val << (nibble * 2));
                i++;
            }
        }
        dst[byte_idx] = byte;
    }
}

void boat_unpack_bits2(const uint8_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (size_t byte_idx = 0; byte_idx < (n + 3) / 4; byte_idx++) {
        uint8_t byte = src[byte_idx];
        for (int nibble = 0; nibble < 4; nibble++) {
            if (i < n) {
                dst[i] = (byte >> (nibble * 2)) & 0x03;
                i++;
            } else {
                break;
            }
        }
    }
}

// FLOAT4 custom floating point format
// Simple representation: sign (1 bit), exponent (3 bits), no mantissa
// Exponent bias: 3 (so exponent range -3 to 4)
boat_float4_t boat_float4_from_float(float f) {
    boat_float4_t f4 = {0};

    if (f == 0.0f) {
        return f4;
    }

    // Extract sign
    if (f < 0) {
        f4.sign = 1;
        f = -f;
    }

    // Normalize to range [1, 2) * 2^exp
    int exp;
    (void)frexpf(f, &exp); // get exponent, normalized value unused
    exp -= 1;

    // Convert to 3-bit exponent with bias 3
    int biased_exp = exp + 3;
    if (biased_exp < 0) biased_exp = 0;
    if (biased_exp > 7) biased_exp = 7;
    f4.exponent = biased_exp & 0x07;

    return f4;
}

float boat_float4_to_float(boat_float4_t f4) {
    if (f4.exponent == 0) {
        return 0.0f;
    }

    int exp = (int)f4.exponent - 3;
    float value = ldexpf(1.0f, exp); // 1 * 2^exp

    if (f4.sign) {
        value = -value;
    }

    return value;
}

void boat_pack_float4(const float* src, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        uint8_t byte = 0;

            boat_float4_t f4 = boat_float4_from_float(src[i]);
            byte |= (f4.sign << 7);
            byte |= (f4.exponent << 4);

        if (i + 1 < n) {
            boat_float4_t f4_2 = boat_float4_from_float(src[i + 1]);
            byte |= (f4_2.sign << 3);
            byte |= (f4_2.exponent);
        }

        dst[i / 2] = byte;
    }
}

void boat_unpack_float4(const uint8_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        uint8_t byte = src[i / 2];

            boat_float4_t f4;
            f4.sign = (byte >> 7) & 1;
            f4.exponent = (byte >> 4) & 0x07;
            dst[i] = boat_float4_to_float(f4);

        if (i + 1 < n) {
            boat_float4_t f4_2;
            f4_2.sign = (byte >> 3) & 1;
            f4_2.exponent = byte & 0x07;
            dst[i + 1] = boat_float4_to_float(f4_2);
        }
    }
}

// FLOAT8 custom floating point format
// Format: sign (1 bit), exponent (4 bits), mantissa (3 bits)
// Exponent bias: 7 (so exponent range -7 to 8)
boat_float8_t boat_float8_from_float(float f) {
    boat_float8_t f8 = {0};

    if (f == 0.0f) {
        return f8;
    }

    // Extract sign
    if (f < 0) {
        f8.sign = 1;
        f = -f;
    }

    // Normalize to range [1, 2) * 2^exp
    int exp;
    float normalized = frexpf(f, &exp); // get normalized value and exponent
    // Adjust to range [1, 2) instead of [0.5, 1)
    normalized *= 2.0f;
    exp -= 1;

    // Convert to 4-bit exponent with bias 7
    int biased_exp = exp + 7;
    if (biased_exp < 0) biased_exp = 0;
    if (biased_exp > 15) biased_exp = 15;
    f8.exponent = biased_exp & 0x0F;

    // Convert mantissa (3 bits from normalized)
    // normalized is in [1, 2), so fractional part is (normalized - 1)
    float frac = normalized - 1.0f;
    f8.mantissa = (uint8_t)(frac * 8.0f) & 0x07; // 3 bits

    return f8;
}

float boat_float8_to_float(boat_float8_t f8) {
    if (f8.exponent == 0 && f8.mantissa == 0) {
        return 0.0f;
    }

    int exp = (int)f8.exponent - 7;
    float mantissa = 1.0f + (float)f8.mantissa / 8.0f;
    float value = ldexpf(mantissa, exp);

    if (f8.sign) {
        value = -value;
    }

    return value;
}

void boat_pack_float8(const float* src, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        boat_float8_t f8 = boat_float8_from_float(src[i]);
        dst[i] = (f8.sign << 7) | (f8.exponent << 3) | f8.mantissa;
    }
}

void boat_unpack_float8(const uint8_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = src[i];
        boat_float8_t f8;
        f8.sign = (byte >> 7) & 1;
        f8.exponent = (byte >> 3) & 0x0F;
        f8.mantissa = byte & 0x07;
        dst[i] = boat_float8_to_float(f8);
    }
}

// Element-wise operations for packed types
// BITS1 addition: logical OR (since values are 0 or 1)
void boat_add_bits1(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes) {
    for (size_t i = 0; i < n_bytes; i++) {
        out[i] = a[i] | b[i];
    }
}

// BITS2 addition: saturated addition (max value 3)
void boat_add_bits2(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes) {
    for (size_t i = 0; i < n_bytes; i++) {
        uint8_t a_byte = a[i];
        uint8_t b_byte = b[i];
        uint8_t result = 0;

        for (int nibble = 0; nibble < 4; nibble++) {
            uint8_t a_val = (a_byte >> (nibble * 2)) & 0x03;
            uint8_t b_val = (b_byte >> (nibble * 2)) & 0x03;
            uint8_t sum = a_val + b_val;
            if (sum > 3) sum = 3;

            result |= (sum << (nibble * 2));
        }

        out[i] = result;
    }
}

// FLOAT4 addition: unpack, add, repack
void boat_add_float4(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes) {
    size_t n_elements = n_bytes * 2; // 2 elements per byte

    // Simple implementation: unpack, add floats, repack
    // For better performance, this should be optimized
    float* a_unpacked = boat_malloc(sizeof(float) * n_elements, BOAT_DEVICE_CPU);
    float* b_unpacked = boat_malloc(sizeof(float) * n_elements, BOAT_DEVICE_CPU);
    float* out_unpacked = boat_malloc(sizeof(float) * n_elements, BOAT_DEVICE_CPU);

    if (a_unpacked && b_unpacked && out_unpacked) {
        boat_unpack_float4(a, a_unpacked, n_elements);
        boat_unpack_float4(b, b_unpacked, n_elements);

        for (size_t i = 0; i < n_elements; i++) {
            out_unpacked[i] = a_unpacked[i] + b_unpacked[i];
        }

        boat_pack_float4(out_unpacked, out, n_elements);
    }

    if (a_unpacked) boat_free(a_unpacked);
    if (b_unpacked) boat_free(b_unpacked);
    if (out_unpacked) boat_free(out_unpacked);
}

// FLOAT8 addition: similar to FLOAT4
void boat_add_float8(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n_bytes) {
    size_t n_elements = n_bytes; // 1 element per byte

    float* a_unpacked = boat_malloc(sizeof(float) * n_elements, BOAT_DEVICE_CPU);
    float* b_unpacked = boat_malloc(sizeof(float) * n_elements, BOAT_DEVICE_CPU);
    float* out_unpacked = boat_malloc(sizeof(float) * n_elements, BOAT_DEVICE_CPU);

    if (a_unpacked && b_unpacked && out_unpacked) {
        boat_unpack_float8(a, a_unpacked, n_elements);
        boat_unpack_float8(b, b_unpacked, n_elements);

        for (size_t i = 0; i < n_elements; i++) {
            out_unpacked[i] = a_unpacked[i] + b_unpacked[i];
        }

        boat_pack_float8(out_unpacked, out, n_elements);
    }

    if (a_unpacked) boat_free(a_unpacked);
    if (b_unpacked) boat_free(b_unpacked);
    if (out_unpacked) boat_free(out_unpacked);
}

// Memory access optimizations for packed types
size_t boat_packed_element_offset(boat_dtype_t dtype, size_t index) {
    switch (dtype) {
        case BOAT_DTYPE_BITS1:
            return index / 8;
        case BOAT_DTYPE_BITS2:
            return index / 4;
        case BOAT_DTYPE_FLOAT4:
            return index / 2;
        case BOAT_DTYPE_FLOAT8:
            return index;
        default:
            return index; // For non-packed types, one element per byte/word
    }
}

uint8_t boat_packed_read_element(const uint8_t* data, boat_dtype_t dtype, size_t index) {
    switch (dtype) {
        case BOAT_DTYPE_BITS1: {
            size_t byte_idx = index / 8;
            int bit = index % 8;
            return (data[byte_idx] >> bit) & 1;
        }
        case BOAT_DTYPE_BITS2: {
            size_t byte_idx = index / 4;
            int nibble = index % 4;
            return (data[byte_idx] >> (nibble * 2)) & 0x03;
        }
        case BOAT_DTYPE_FLOAT4: {
            size_t byte_idx = index / 2;
            if (index % 2 == 0) {
                return (data[byte_idx] >> 4) & 0x0F;
            } else {
                return data[byte_idx] & 0x0F;
            }
        }
        case BOAT_DTYPE_FLOAT8:
            return data[index];
        default:
            return 0;
    }
}

void boat_packed_write_element(uint8_t* data, boat_dtype_t dtype, size_t index, uint8_t value) {
    switch (dtype) {
        case BOAT_DTYPE_BITS1: {
            size_t byte_idx = index / 8;
            int bit = index % 8;
            uint8_t mask = ~(1 << bit);
            data[byte_idx] = (data[byte_idx] & mask) | ((value & 1) << bit);
            break;
        }
        case BOAT_DTYPE_BITS2: {
            size_t byte_idx = index / 4;
            int nibble = index % 4;
            uint8_t mask = ~(0x03 << (nibble * 2));
            data[byte_idx] = (data[byte_idx] & mask) | ((value & 0x03) << (nibble * 2));
            break;
        }
        case BOAT_DTYPE_FLOAT4: {
            size_t byte_idx = index / 2;
            if (index % 2 == 0) {
                data[byte_idx] = (data[byte_idx] & 0x0F) | ((value & 0x0F) << 4);
            } else {
                data[byte_idx] = (data[byte_idx] & 0xF0) | (value & 0x0F);
            }
            break;
        }
        case BOAT_DTYPE_FLOAT8:
            data[index] = value;
            break;
        default:
            break;
    }
}