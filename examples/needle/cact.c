// cact.c - Parser and dequantizer for the Needle .cact deployment blob.
//
// Implements the byte layout documented in needle/model/export.py:
//   - 120-byte little-endian header ("<29If")
//   - codebook_len * f32 Lloyd-Max unit-sphere codebooks (cb2 | cb3 | cb4)
//   - num_tensors directory records ("<BBHIIIIQQII", 48 bytes each)
//   - 64-byte aligned tensor blobs
// CQ tensors: [out, in] with in padded to a multiple of group_size; per row
// LSB-packed index bitstream followed by per-group FP16 L2 norms.
// Reconstruct per group: w = (codebook[idx] * norm) @ H, H = normalized
// Walsh-Hadamard(group). The fast Hadamard transform computes the product.

#include "cact.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// In-place fast Walsh-Hadamard transform (unnormalized).
static void fwht(float* a, int n) {
    for (int m = 1; m < n; m <<= 1) {
        for (int i = 0; i < n; i += m << 1) {
            for (int j = 0; j < m; j++) {
                float x = a[i + j];
                float y = a[i + j + m];
                a[i + j] = x + y;
                a[i + j + m] = x - y;
            }
        }
    }
}

static uint16_t load_u16_le(const uint8_t* p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t load_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t load_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t)p[i] << (8 * i);
    }
    return v;
}

static float load_f32_le(const uint8_t* p) {
    uint32_t v = load_u32_le(p);
    float f;
    memcpy(&f, &v, sizeof(f));
    return f;
}

// IEEE-754 half -> float (little-endian input, host-endian output).
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31; // +/- 0
        } else {
            // Subnormal: normalize.
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FFu;
            bits = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (man << 13); // inf/nan
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int needle_cact_open(needle_cact_t* cact, const char* path) {
    memset(cact, 0, sizeof(*cact));
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[needle] cannot open %s\n", path);
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    long size = ftell(fp);
    if (size < 0 || size < 120) {
        fclose(fp);
        return -1;
    }
    rewind(fp);
    cact->owned = (uint8_t*)malloc((size_t)size);
    if (!cact->owned) {
        fclose(fp);
        return -1;
    }
    if (fread(cact->owned, 1, (size_t)size, fp) != (size_t)size) {
        fclose(fp);
        free(cact->owned);
        cact->owned = NULL;
        return -1;
    }
    fclose(fp);

    const uint8_t* p = cact->owned;
    needle_cact_header_t* h = &cact->hdr;
    h->tag = load_u32_le(p + 0);
    h->num_tensors = load_u32_le(p + 4);
    h->codebook_len = load_u32_le(p + 8);
    h->kv_window = load_u32_le(p + 12);
    h->kv_bits = load_u32_le(p + 16);
    h->vocab = load_u32_le(p + 20);
    h->d_model = load_u32_le(p + 24);
    h->num_heads = load_u32_le(p + 28);
    h->num_kv_heads = load_u32_le(p + 32);
    h->num_layers = load_u32_le(p + 36);
    h->head_dim = load_u32_le(p + 40);
    h->max_seq_len = load_u32_le(p + 44);
    h->hada_n = load_u32_le(p + 48);
    h->mhc_lanes = load_u32_le(p + 52);
    h->engram_slots = load_u32_le(p + 56);
    h->engram_sub_dim = load_u32_le(p + 60);
    h->num_engram_tables = load_u32_le(p + 64);
    h->engram_conv_taps = load_u32_le(p + 68);
    h->engram_conv_dilation = load_u32_le(p + 72);
    h->num_engram_orders = load_u32_le(p + 76);
    for (int i = 0; i < 4; i++) {
        h->engram_orders[i] = load_u32_le(p + 80 + 4 * i);
    }
    h->num_engram_sites = load_u32_le(p + 96);
    for (int i = 0; i < 4; i++) {
        h->engram_sites[i] = load_u32_le(p + 100 + 4 * i);
    }
    h->rope_theta = load_f32_le(p + 116);

    if (h->tag != NEEDLE_CACT_TAG) {
        fprintf(stderr, "[needle] bad cact tag 0x%08x\n", h->tag);
        needle_cact_close(cact);
        return -1;
    }
    if (h->num_tensors == 0 || h->num_tensors > 100000 || h->codebook_len > 65536) {
        fprintf(stderr, "[needle] implausible header\n");
        needle_cact_close(cact);
        return -1;
    }

    // Codebook: cb2[4] | cb3[8] | cb4[16].
    cact->codebook = (float*)malloc((size_t)h->codebook_len * sizeof(float));
    if (!cact->codebook) {
        needle_cact_close(cact);
        return -1;
    }
    const uint8_t* cb = p + 120;
    for (uint32_t i = 0; i < h->codebook_len; i++) {
        cact->codebook[i] = load_f32_le(cb + 4 * i);
    }

    // Directory.
    cact->recs = (needle_cact_rec_t*)malloc((size_t)h->num_tensors * sizeof(needle_cact_rec_t));
    if (!cact->recs) {
        needle_cact_close(cact);
        return -1;
    }
    const uint8_t* d = cb + 4 * (size_t)h->codebook_len;
    for (uint32_t i = 0; i < h->num_tensors; i++) {
        needle_cact_rec_t* r = &cact->recs[i];
        r->dtype = d[0];
        r->ndim = d[1];
        r->pad = load_u16_le(d + 2);
        for (int k = 0; k < 4; k++) {
            r->shape[k] = load_u32_le(d + 4 + 4 * k);
        }
        r->offset = load_u64_le(d + 20);
        r->nbytes = load_u64_le(d + 28);
        r->group_size = load_u32_le(d + 36);
        r->bits = load_u32_le(d + 40);
        d += 44; // <BBHIIIIQQII
    }
    return 0;
}

void needle_cact_close(needle_cact_t* cact) {
    free(cact->recs);
    free(cact->codebook);
    free(cact->owned);
    memset(cact, 0, sizeof(*cact));
}

static size_t rec_nelements(const needle_cact_rec_t* r) {
    size_t n = 1;
    for (int i = 0; i < (int)r->ndim; i++) {
        n *= r->shape[i];
    }
    return n;
}

// Dequantize one CQ tensor into `out` (fp32 [out, in]).
static int dequant_cq(const needle_cact_t* cact, const needle_cact_rec_t* r, float* out,
                      size_t out_cap) {
    const uint32_t out_rows = r->shape[0];
    const uint32_t in_dim = r->shape[1];
    const uint32_t g = r->group_size ? r->group_size : 128u;
    const uint32_t bits = r->bits;
    const uint32_t in_pad = ((in_dim + g - 1) / g) * g;
    if ((size_t)out_rows * in_dim > out_cap) {
        return -1;
    }
    if (r->nbytes == 0 || g == 0) {
        return -1;
    }

    const uint8_t* blob = cact->owned + r->offset;
    size_t per_row;
    if (bits == NEEDLE_TERNARY_RECORD_BITS) {
        per_row = (size_t)in_pad * 2 / 8; // 4 crumb-packed 2-bit fields/byte
    } else {
        per_row = (size_t)in_pad * bits / 8;
    }
    if (per_row * out_rows > r->nbytes) {
        return -1;
    }
    const uint8_t* packed = blob;
    const uint8_t* norms = blob + per_row * out_rows;
    const uint32_t groups_per_row = in_pad / g;
    if ((size_t)groups_per_row * out_rows * 2 > r->nbytes - per_row * out_rows) {
        return -1;
    }

    const float* codebook;
    uint32_t levels;
    float c_over_sqrtg; // ternary centroid magnitude / sqrt(group), analytic
    if (bits == 2) {
        codebook = cact->codebook;
        levels = 4;
        c_over_sqrtg = 0.0f;
    } else if (bits == 3) {
        codebook = cact->codebook + 4;
        levels = 8;
        c_over_sqrtg = 0.0f;
    } else if (bits == 4) {
        codebook = cact->codebook + 12;
        levels = 16;
        c_over_sqrtg = 0.0f;
    } else if (bits == NEEDLE_TERNARY_RECORD_BITS) {
        codebook = NULL;
        levels = 3;
        c_over_sqrtg = 1.2240064f / sqrtf((float)g); // 3-level Lloyd-Max centroid
    } else {
        return -1;
    }

    float* rowbuf = (float*)malloc((size_t)in_pad * sizeof(float));
    if (!rowbuf) {
        return -1;
    }
    for (uint32_t rw = 0; rw < out_rows; rw++) {
        // Unpack LSB-first index stream for this row.
        const uint8_t* prow = packed + (size_t)rw * per_row;
        for (uint32_t c8 = 0; c8 < in_pad / 8; c8++) {
            uint64_t word = 0;
            for (uint32_t b = 0; b < bits; b++) {
                word |= (uint64_t)prow[(size_t)c8 * bits + b] << (8 * b);
            }
            for (uint32_t i = 0; i < 8; i++) {
                uint32_t idx = (uint32_t)((word >> (i * bits)) & ((1u << bits) - 1u));
                if (bits == NEEDLE_TERNARY_RECORD_BITS) {
                    // crumb 3,0,1 -> trit 0,1,2 ; sign-extend 2-bit -> -1,0,+1
                    int32_t trit = (int32_t)(idx << 30) >> 30; // sign extend 2 bits
                    rowbuf[(size_t)c8 * 8 + i] = (float)trit * c_over_sqrtg;
                } else {
                    if (idx >= levels) {
                        free(rowbuf);
                        return -1;
                    }
                    rowbuf[(size_t)c8 * 8 + i] = codebook[idx];
                }
            }
        }
        // Dequantize per group: w = (cb[idx]*norm) @ H = fwht(rot)/sqrt(g).
        for (uint32_t q = 0; q < groups_per_row; q++) {
            float norm = f16_to_f32(load_u16_le(norms + 2 * ((size_t)rw * groups_per_row + q)));
            float* grp = rowbuf + (size_t)q * g;
            for (uint32_t j = 0; j < g; j++) {
                grp[j] *= norm;
            }
            fwht(grp, (int)g);
            float inv = 1.0f / sqrtf((float)g);
            for (uint32_t j = 0; j < g; j++) {
                grp[j] *= inv;
            }
        }
        memcpy(out + (size_t)rw * in_dim, rowbuf, (size_t)in_dim * sizeof(float));
    }
    free(rowbuf);
    return (int)((size_t)out_rows * in_dim);
}

int needle_cact_tensor_f32(const needle_cact_t* cact, uint32_t idx, float* out, size_t out_cap) {
    if (idx >= cact->hdr.num_tensors) {
        return -1;
    }
    const needle_cact_rec_t* r = &cact->recs[idx];
    const uint8_t* blob = cact->owned + r->offset;
    size_t n = rec_nelements(r);
    if (n > out_cap) {
        return -1;
    }
    if (r->dtype == NEEDLE_ND_CQ) {
        return dequant_cq(cact, r, out, out_cap);
    }
    if (r->dtype == NEEDLE_ND_FP16) {
        for (size_t i = 0; i < n; i++) {
            out[i] = f16_to_f32(load_u16_le(blob + 2 * i));
        }
        return (int)n;
    }
    if (r->dtype == NEEDLE_ND_FP32) {
        if (r->nbytes < n * 4) {
            return -1;
        }
        for (size_t i = 0; i < n; i++) {
            out[i] = load_f32_le(blob + 4 * i);
        }
        return (int)n;
    }
    return -1; // RAW
}

int needle_cact_tensor_raw(const needle_cact_t* cact, uint32_t idx, const uint8_t** data,
                           size_t* nbytes) {
    if (idx >= cact->hdr.num_tensors) {
        return -1;
    }
    const needle_cact_rec_t* r = &cact->recs[idx];
    *data = cact->owned + r->offset;
    *nbytes = (size_t)r->nbytes;
    return 0;
}

uint32_t needle_cact_index_layer(const needle_cact_t* cact, uint32_t layer) {
    (void)cact;
    return 1 + layer * 14;
}

uint32_t needle_cact_index_mhc_scalar(const needle_cact_t* cact, uint32_t which) {
    return 1 + cact->hdr.num_layers * 14 + which; // which in [0,6)
}

uint32_t needle_cact_index_mhc_phi(const needle_cact_t* cact, uint32_t which) {
    return 1 + cact->hdr.num_layers * 14 + 6 + which; // which in [0,3)
}

uint32_t needle_cact_index_engram(const needle_cact_t* cact, uint32_t site) {
    return 1 + cact->hdr.num_layers * 14 + 9 + site * 4;
}

uint32_t needle_cact_index_final_norm(const needle_cact_t* cact) {
    return 1 + cact->hdr.num_layers * 14 + 9 + cact->hdr.num_engram_sites * 4;
}

uint32_t needle_cact_index_tokenizer(const needle_cact_t* cact) {
    for (uint32_t i = 0; i < cact->hdr.num_tensors; i++) {
        if (cact->recs[i].dtype == NEEDLE_ND_RAW) {
            return i;
        }
    }
    return cact->hdr.num_tensors; // invalid sentinel
}
