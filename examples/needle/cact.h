// cact.h - Parser for the Needle .cact deployment blob.
//
// The .cact format is documented in needle/model/export.py of the Cactus
// needle repo. A blob is: a fixed 120-byte geometry header, a nameless
// directory of 48-byte tensor records, 64-byte-aligned tensor blobs, and a
// RAW tokenizer blob. CQ tensors are stored pre-transposed [out, in] with
// LSB-packed codebook indices and per-group L2 norms.

#ifndef NEEDLE_CACT_H
#define NEEDLE_CACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEEDLE_CACT_TAG 0x05E12A83u
#define NEEDLE_CACT_ALIGN 64u

// Tensor dtypes in the directory records.
#define NEEDLE_ND_FP16 1
#define NEEDLE_ND_FP32 2
#define NEEDLE_ND_CQ 3
#define NEEDLE_ND_RAW 4

// Bits recorded for ternary tensors (4 crumb-packed 2-bit fields per byte).
#define NEEDLE_TERNARY_RECORD_BITS 5

typedef struct {
    uint32_t tag;
    uint32_t num_tensors;
    uint32_t codebook_len;
    uint32_t kv_window;
    uint32_t kv_bits;
    uint32_t vocab;
    uint32_t d_model;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t num_layers;
    uint32_t head_dim;
    uint32_t max_seq_len;
    uint32_t hada_n;
    uint32_t mhc_lanes;
    uint32_t engram_slots;
    uint32_t engram_sub_dim;
    uint32_t num_engram_tables;
    uint32_t engram_conv_taps;
    uint32_t engram_conv_dilation;
    uint32_t num_engram_orders;
    uint32_t engram_orders[4];
    uint32_t num_engram_sites;
    uint32_t engram_sites[4];
    float rope_theta;
} needle_cact_header_t;

typedef struct {
    uint8_t dtype;
    uint8_t ndim;
    uint16_t pad;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t nbytes;
    uint32_t group_size;
    uint32_t bits;
} needle_cact_rec_t;

typedef struct {
    needle_cact_header_t hdr;
    needle_cact_rec_t* recs;
    float* codebook;   // codebook_len floats: cb2[4] | cb3[8] | cb4[16]
    uint8_t* owned;    // owning buffer (whole file contents)
} needle_cact_t;

// Load the blob at `path`. Returns 0 on success, -1 on failure.
int needle_cact_open(needle_cact_t* cact, const char* path);
void needle_cact_close(needle_cact_t* cact);

// Dequantize tensor `idx` into `out` (fp32, row-major, logical shape).
// `out` must hold shape[0]*...*shape[ndim-1] floats. Returns element count
// on success, -1 on error. RAW tensors are not dequantizable.
int needle_cact_tensor_f32(const needle_cact_t* cact, uint32_t idx,
                           float* out, size_t out_cap);

// Pointer to a RAW tensor blob (e.g. the tokenizer). Returns 0 on success.
int needle_cact_tensor_raw(const needle_cact_t* cact, uint32_t idx,
                           const uint8_t** data, size_t* nbytes);

// Fixed-position tensor index helpers derived from the header geometry.
// Each layer occupies 14 consecutive tensors starting at index_layer(layer).
uint32_t needle_cact_index_layer(const needle_cact_t* cact, uint32_t layer);
uint32_t needle_cact_index_mhc_scalar(const needle_cact_t* cact, uint32_t which);
uint32_t needle_cact_index_mhc_phi(const needle_cact_t* cact, uint32_t which);
uint32_t needle_cact_index_engram(const needle_cact_t* cact, uint32_t site);
uint32_t needle_cact_index_final_norm(const needle_cact_t* cact);
uint32_t needle_cact_index_tokenizer(const needle_cact_t* cact);

#ifdef __cplusplus
}
#endif

#endif // NEEDLE_CACT_H
