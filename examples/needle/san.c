// san.c - Simple Attention Network inference executor for the Needle .cact
// deployment weights.
//
// Reference: needle/model/decode.py (generate_cached / _forward_cached) of
// the Cactus needle repo. Numerics follow the JAX decode path exactly:
// fp32 activations, ZCRMSNorm with (1+scale), split-half RoPE, Walsh-Hadamard
// MLP via the fast transform, engram hashed n-gram tables with conv taps, and
// 4-lane MHC routing with Sinkhorn normalization.

#include <stdio.h>
#include "san.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include "tokenizer.h"

#define SAN_EPS 1e-6f
#define SAN_SINKHORN_ITERS 20

struct needle_model {
    // Geometry (copied from the cact header).
    uint32_t vocab, d_model, num_heads, num_kv_heads, num_layers, head_dim;
    uint32_t max_seq_len, kv_window, hada_n, mhc_lanes;
    uint32_t engram_slots, engram_sub_dim, num_engram_tables;
    uint32_t engram_conv_taps, engram_conv_dilation;
    uint32_t engram_orders[4], num_engram_orders;
    uint32_t engram_sites[4], num_engram_sites;
    float rope_theta;

    // Tied embedding [vocab, d_model].
    float* embedding;

    // Per-layer tensors (rows = layer).
    float* norm_in;    // [L, d]
    float* q_proj;     // [L, d, d]
    float* k_proj;     // [L, d, kv]
    float* v_proj;     // [L, d, kv]
    float* q_norm;     // [L, hd]
    float* k_norm;     // [L, hd]
    float* gate_proj;  // [L, d, d]
    float* out_proj;   // [L, d, d]
    float* post_norm;  // [L, d]
    float* attn_gate;  // [L]
    float* pre_hada;   // [L, d]
    float* d1;         // [L, d]
    float* d2;         // [L, d]
    float* d3;         // [L, d]

    // MHC routing.
    float* mhc_a_pre;    // [L]
    float* mhc_a_post;   // [L]
    float* mhc_a_res;    // [L]
    float* mhc_b_pre;    // [L, n]
    float* mhc_b_post;   // [L, n]
    float* mhc_b_res;    // [L, n, n]
    float* mhc_phi_pre;  // [L*n, n*d]
    float* mhc_phi_post; // [L*n, n*d]
    float* mhc_phi_res;  // [L*n, n*d]

    // Engram sites.
    float* eg_tables[2];  // [n_tables*slots, sub]
    float* eg_key[2];     // [d, d]
    float* eg_value[2];   // [d, d]
    float* eg_taps[2];    // [taps, d]

    float* final_norm;  // [d]
};

const char* needle_model_engine_version(void) { return "needle2-san 0.1 (boat)"; }

static void* xcalloc(size_t n, size_t sz) {
    if (n && sz && n > (size_t)-1 / sz) return NULL;
    return calloc(n, sz);
}

// Load one cact tensor into `dst` (already sized by the caller).
static int load_tensor(needle_model_t* m, const needle_cact_t* c, uint32_t idx,
                       float* dst, size_t n) {
    (void)m;
    int got = needle_cact_tensor_f32(c, idx, dst, n);
    return got == (int)n ? 0 : -1;
}

needle_model_t* needle_model_load(const needle_cact_t* cact) {
    needle_model_t* m = (needle_model_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    const needle_cact_header_t* h = &cact->hdr;

    m->vocab = h->vocab;
    m->d_model = h->d_model;
    m->num_heads = h->num_heads;
    m->num_kv_heads = h->num_kv_heads;
    m->num_layers = h->num_layers;
    m->head_dim = h->head_dim;
    m->max_seq_len = h->max_seq_len;
    m->kv_window = h->kv_window ? h->kv_window : h->max_seq_len;
    m->hada_n = h->hada_n;
    m->mhc_lanes = h->mhc_lanes;
    m->engram_slots = h->engram_slots;
    m->engram_sub_dim = h->engram_sub_dim;
    m->num_engram_tables = h->num_engram_tables;
    m->engram_conv_taps = h->engram_conv_taps;
    m->engram_conv_dilation = h->engram_conv_dilation;
    m->num_engram_orders = h->num_engram_orders;
    for (uint32_t i = 0; i < 4; i++) m->engram_orders[i] = h->engram_orders[i];
    m->num_engram_sites = h->num_engram_sites;
    for (uint32_t i = 0; i < 4; i++) m->engram_sites[i] = h->engram_sites[i];
    m->rope_theta = h->rope_theta;

    const uint32_t L = m->num_layers;
    const uint32_t d = m->d_model;
    const uint32_t kv = m->num_kv_heads * m->head_dim;
    const uint32_t n = m->mhc_lanes;
    const uint32_t nC = n * d;

    // Allocate.
    m->embedding = xcalloc((size_t)m->vocab * d, sizeof(float));
    m->norm_in = xcalloc((size_t)L * d, sizeof(float));
    m->q_proj = xcalloc((size_t)L * d * d, sizeof(float));
    m->k_proj = xcalloc((size_t)L * d * kv, sizeof(float));
    m->v_proj = xcalloc((size_t)L * d * kv, sizeof(float));
    m->q_norm = xcalloc((size_t)L * m->head_dim, sizeof(float));
    m->k_norm = xcalloc((size_t)L * m->head_dim, sizeof(float));
    m->gate_proj = xcalloc((size_t)L * d * d, sizeof(float));
    m->out_proj = xcalloc((size_t)L * d * d, sizeof(float));
    m->post_norm = xcalloc((size_t)L * d, sizeof(float));
    m->attn_gate = xcalloc(L, sizeof(float));
    m->pre_hada = xcalloc((size_t)L * d, sizeof(float));
    m->d1 = xcalloc((size_t)L * d, sizeof(float));
    m->d2 = xcalloc((size_t)L * d, sizeof(float));
    m->d3 = xcalloc((size_t)L * d, sizeof(float));
    m->mhc_a_pre = xcalloc(L, sizeof(float));
    m->mhc_a_post = xcalloc(L, sizeof(float));
    m->mhc_a_res = xcalloc(L, sizeof(float));
    m->mhc_b_pre = xcalloc((size_t)L * n, sizeof(float));
    m->mhc_b_post = xcalloc((size_t)L * n, sizeof(float));
    m->mhc_b_res = xcalloc((size_t)L * n * n, sizeof(float));
    m->mhc_phi_pre = xcalloc((size_t)L * n * nC, sizeof(float));
    m->mhc_phi_post = xcalloc((size_t)L * n * nC, sizeof(float));
    m->mhc_phi_res = xcalloc((size_t)L * n * n * nC, sizeof(float));
    m->final_norm = xcalloc(d, sizeof(float));
    if (!m->embedding || !m->norm_in || !m->q_proj || !m->k_proj || !m->v_proj ||
        !m->q_norm || !m->k_norm || !m->gate_proj || !m->out_proj || !m->post_norm ||
        !m->attn_gate || !m->pre_hada || !m->d1 || !m->d2 || !m->d3 || !m->mhc_a_pre ||
        !m->mhc_a_post || !m->mhc_a_res || !m->mhc_b_pre || !m->mhc_b_post ||
        !m->mhc_b_res || !m->mhc_phi_pre || !m->mhc_phi_post || !m->mhc_phi_res ||
        !m->final_norm) {
        needle_model_free(m);
        return NULL;
    }

    int rc = 0;
    uint32_t idx = 0;

    // embedding.
    rc |= load_tensor(m, cact, idx++, m->embedding, (size_t)m->vocab * d);
    // Layers.
    for (uint32_t li = 0; li < L; li++) {
        float* w = m->norm_in + (size_t)li * d;
        rc |= load_tensor(m, cact, idx++, w, d);
        w = m->q_proj + (size_t)li * d * d;
        rc |= load_tensor(m, cact, idx++, w, (size_t)d * d);
        w = m->k_proj + (size_t)li * d * kv;
        rc |= load_tensor(m, cact, idx++, w, (size_t)d * kv);
        w = m->v_proj + (size_t)li * d * kv;
        rc |= load_tensor(m, cact, idx++, w, (size_t)d * kv);
        w = m->q_norm + (size_t)li * m->head_dim;
        rc |= load_tensor(m, cact, idx++, w, m->head_dim);
        w = m->k_norm + (size_t)li * m->head_dim;
        rc |= load_tensor(m, cact, idx++, w, m->head_dim);
        w = m->gate_proj + (size_t)li * d * d;
        rc |= load_tensor(m, cact, idx++, w, (size_t)d * d);
        w = m->out_proj + (size_t)li * d * d;
        rc |= load_tensor(m, cact, idx++, w, (size_t)d * d);
        w = m->post_norm + (size_t)li * d;
        rc |= load_tensor(m, cact, idx++, w, d);
        rc |= load_tensor(m, cact, idx++, m->attn_gate + li, 1);
        w = m->pre_hada + (size_t)li * d;
        rc |= load_tensor(m, cact, idx++, w, d);
        w = m->d1 + (size_t)li * d;
        rc |= load_tensor(m, cact, idx++, w, d);
        w = m->d2 + (size_t)li * d;
        rc |= load_tensor(m, cact, idx++, w, d);
        w = m->d3 + (size_t)li * d;
        rc |= load_tensor(m, cact, idx++, w, d);
    }
    // MHC scalars.
    rc |= load_tensor(m, cact, idx++, m->mhc_a_pre, L);
    rc |= load_tensor(m, cact, idx++, m->mhc_a_post, L);
    rc |= load_tensor(m, cact, idx++, m->mhc_a_res, L);
    rc |= load_tensor(m, cact, idx++, m->mhc_b_pre, (size_t)L * n);
    rc |= load_tensor(m, cact, idx++, m->mhc_b_post, (size_t)L * n);
    rc |= load_tensor(m, cact, idx++, m->mhc_b_res, (size_t)L * n * n);
    rc |= load_tensor(m, cact, idx++, m->mhc_phi_pre, (size_t)L * n * nC);
    rc |= load_tensor(m, cact, idx++, m->mhc_phi_post, (size_t)L * n * nC);
    rc |= load_tensor(m, cact, idx++, m->mhc_phi_res, (size_t)L * n * n * nC);
    // Engram sites.
    for (uint32_t s = 0; s < m->num_engram_sites; s++) {
        uint32_t n_tab = m->num_engram_tables * m->engram_slots;
        m->eg_tables[s] = xcalloc((size_t)n_tab * m->engram_sub_dim, sizeof(float));
        m->eg_key[s] = xcalloc((size_t)d * d, sizeof(float));
        m->eg_value[s] = xcalloc((size_t)d * d, sizeof(float));
        m->eg_taps[s] = xcalloc((size_t)m->engram_conv_taps * d, sizeof(float));
        if (!m->eg_tables[s] || !m->eg_key[s] || !m->eg_value[s] || !m->eg_taps[s]) {
            needle_model_free(m);
            return NULL;
        }
        rc |= load_tensor(m, cact, idx++, m->eg_tables[s], (size_t)n_tab * m->engram_sub_dim);
        rc |= load_tensor(m, cact, idx++, m->eg_key[s], (size_t)d * d);
        rc |= load_tensor(m, cact, idx++, m->eg_value[s], (size_t)d * d);
        rc |= load_tensor(m, cact, idx++, m->eg_taps[s], (size_t)m->engram_conv_taps * d);
    }
    // Final norm.
    rc |= load_tensor(m, cact, idx++, m->final_norm, d);

    if (rc != 0) {
        needle_model_free(m);
        return NULL;
    }
    return m;
}

void needle_model_free(needle_model_t* m) {
    if (!m) return;
    free(m->embedding);
    free(m->norm_in);
    free(m->q_proj);
    free(m->k_proj);
    free(m->v_proj);
    free(m->q_norm);
    free(m->k_norm);
    free(m->gate_proj);
    free(m->out_proj);
    free(m->post_norm);
    free(m->attn_gate);
    free(m->pre_hada);
    free(m->d1);
    free(m->d2);
    free(m->d3);
    free(m->mhc_a_pre);
    free(m->mhc_a_post);
    free(m->mhc_a_res);
    free(m->mhc_b_pre);
    free(m->mhc_b_post);
    free(m->mhc_b_res);
    free(m->mhc_phi_pre);
    free(m->mhc_phi_post);
    free(m->mhc_phi_res);
    free(m->final_norm);
    for (int s = 0; s < 2; s++) {
        free(m->eg_tables[s]);
        free(m->eg_key[s]);
        free(m->eg_value[s]);
        free(m->eg_taps[s]);
    }
    free(m);
}

// --- small math helpers ---------------------------------------------------

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x) { return x * sigmoidf(x); }

static float logsumexp4(const float* v) {
    float m = v[0];
    for (int i = 1; i < 4; i++) {
        if (v[i] > m) m = v[i];
    }
    float s = 0.0f;
    for (int i = 0; i < 4; i++) s += expf(v[i] - m);
    return m + logf(s);
}

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

// y[s][o] = sum_d x[s][d] * W[o][d], W is [O, D] row-major.
static void gemm_xwt(const float* x, const float* W, int S, int D, int O, float* y) {
    for (int s = 0; s < S; s++) {
        const float* xr = x + (size_t)s * D;
        float* yr = y + (size_t)s * O;
        for (int o = 0; o < O; o++) {
            const float* wr = W + (size_t)o * D;
            float acc = 0.0f;
            for (int d = 0; d < D; d++) acc += xr[d] * wr[d];
            yr[o] = acc;
        }
    }
}

// ZCRMSNorm: out = (1 + scale) * x / rms(x, eps). `x` is row-major [S, D],
// scale is [D].
static void zcrms(float* x, const float* scale, int S, int D) {
    for (int s = 0; s < S; s++) {
        float* xr = x + (size_t)s * D;
        float ss = 0.0f;
        for (int d = 0; d < D; d++) ss += xr[d] * xr[d];
        float rms = sqrtf(ss / (float)D + SAN_EPS);
        for (int d = 0; d < D; d++) xr[d] = (1.0f + scale[d]) * xr[d] / rms;
    }
}

// ZCRMSNorm per head: rows are [S, H*hd], scale is [hd] (shared across heads).
static void zcrms_heads(float* x, const float* scale, int S, int H, int hd) {
    for (int s = 0; s < S; s++) {
        float* xr = x + (size_t)s * H * hd;
        for (int h = 0; h < H; h++) {
            float* hr = xr + (size_t)h * hd;
            float ss = 0.0f;
            for (int i = 0; i < hd; i++) ss += hr[i] * hr[i];
            float rms = sqrtf(ss / (float)hd + SAN_EPS);
            for (int i = 0; i < hd; i++) hr[i] = (1.0f + scale[i]) * hr[i] / rms;
        }
    }
}

// RoPE (split-half convention) applied in place to q [S, H, hd] and
// k [S, KV, hd] for positions pos..pos+S-1. cos/sin are [max_len, hd/2].
static void apply_rope(float* q, int H, float* k, int KV, int S, int hd, int pos,
                       const float* cos_t, const float* sin_t) {
    int half = hd / 2;
    for (int s = 0; s < S; s++) {
        int t = pos + s;
        const float* c = cos_t + (size_t)t * half;
        const float* sn = sin_t + (size_t)t * half;
        for (int h = 0; h < H; h++) {
            float* qr = q + ((size_t)s * H + h) * hd;
            for (int i = 0; i < half; i++) {
                float x1 = qr[i];
                float x2 = qr[i + half];
                qr[i] = x1 * c[i] - x2 * sn[i];
                qr[i + half] = x2 * c[i] + x1 * sn[i];
            }
        }
        for (int h = 0; h < KV; h++) {
            float* kr = k + ((size_t)s * KV + h) * hd;
            for (int i = 0; i < half; i++) {
                float x1 = kr[i];
                float x2 = kr[i + half];
                kr[i] = x1 * c[i] - x2 * sn[i];
                kr[i + half] = x2 * c[i] + x1 * sn[i];
            }
        }
    }
}

// Sinkhorn (20 iters) on a [4,4] matrix (log-space, row then col normalize).
static void sinkhorn44(const float* in, float* out) {
    float logk[16];
    for (int i = 0; i < 16; i++) logk[i] = in[i];
    for (int it = 0; it < SAN_SINKHORN_ITERS; it++) {
        for (int r = 0; r < 4; r++) {
            float m = logsumexp4(logk + 4 * r);
            for (int c = 0; c < 4; c++) logk[4 * r + c] -= m;
        }
        for (int c = 0; c < 4; c++) {
            float col[4];
            for (int r = 0; r < 4; r++) col[r] = logk[4 * r + c];
            float m = logsumexp4(col);
            for (int r = 0; r < 4; r++) logk[4 * r + c] -= m;
        }
    }
    for (int i = 0; i < 16; i++) out[i] = expf(logk[i]);
}

// --- engram ---------------------------------------------------------------

// Compute engram keys/values for S output positions starting at `pos`.
// `hist` holds the full token history (zeros beyond the current length).
static void engram_kv(const needle_model_t* m, const float* tables, const float* key_proj,
                      const float* value_proj, const float* taps, const int32_t* hist,
                      uint32_t pos, uint32_t S, float* ek, float* ev) {
    const uint32_t sub = m->engram_sub_dim;
    const uint32_t slots = m->engram_slots;
    const uint32_t n_tables = m->num_engram_tables;
    const uint32_t d = m->d_model;
    const uint32_t taps_n = m->engram_conv_taps;
    const uint32_t dil = m->engram_conv_dilation;

    // Gather fetched rows into a [S, n_tables*sub] buffer.
    size_t Sd = (size_t)S * (n_tables * sub);
    float* e = (float*)xcalloc(Sd ? Sd : 1, sizeof(float));
    if (!e) return;

    for (uint32_t t = 0; t < S; t++) {
        float* er = e + (size_t)t * (n_tables * sub);
        uint32_t ti = 0;
        for (uint32_t oi = 0; oi < m->num_engram_orders; oi++) {
            uint32_t order = m->engram_orders[oi];
            for (uint32_t h = 0; h < n_tables / m->num_engram_orders; h++) {
                uint32_t table = ti;
                ti++;
                // Hash the o-gram ending at hist[pos + t].
                uint32_t seed = (0x9E3779B9u * (table + 1)) & 0xFFFFFFFFu;
                uint32_t acc = seed;
                for (uint32_t j = 0; j < order; j++) {
                    uint32_t idx_pos = pos + t - j;
                    uint32_t tok = 0;
                    if ((int32_t)idx_pos >= 0) tok = (uint32_t)hist[idx_pos];
                    acc = (acc ^ tok) * 0x01000193u;
                }
                acc ^= acc >> 15;
                uint32_t slot = acc % slots;
                // Active only when the full n-gram is within real history:
                // the oldest token of the o-gram (pos + t - (o-1)) must be >= 0.
                if (pos + t < order - 1) continue;
                const float* row = tables + ((size_t)table * slots + slot) * sub;
                for (uint32_t c = 0; c < sub; c++) {
                    er[(size_t)table * sub + c] = row[c];
                }
            }
        }
    }

    // k = e @ key_proj, v = e @ value_proj.
    size_t kd = (size_t)S * d;
    float* k = (float*)xcalloc(kd ? kd : 1, sizeof(float));
    float* v = (float*)xcalloc(kd ? kd : 1, sizeof(float));
    if (!k || !v) {
        free(e);
        free(k);
        free(v);
        return;
    }
    gemm_xwt(e, key_proj, (int)S, (int)(n_tables * sub), (int)d, k);
    gemm_xwt(e, value_proj, (int)S, (int)(n_tables * sub), (int)d, v);
    free(e);

    // Conv taps: v_out[t] = sum_j taps[j] * v[t - j*dil] (zeros out of range).
    for (uint32_t t = 0; t < S; t++) {
        float* vout = ev + (size_t)t * d;
        const float* krow = k + (size_t)t * d;
        memcpy(ek + (size_t)t * d, krow, d * sizeof(float));
        for (uint32_t c = 0; c < d; c++) vout[c] = 0.0f;
        for (uint32_t j = 0; j < taps_n; j++) {
            uint32_t back = j * dil;
            if (t < back) continue;
            const float* vrow = v + (size_t)(t - back) * d;
            const float* tap = taps + (size_t)j * d;
            for (uint32_t c = 0; c < d; c++) {
                vout[c] += tap[c] * vrow[c];
            }
        }
    }
    free(k);
    free(v);
}

// --- forward --------------------------------------------------------------

// Forward over S tokens at absolute positions pos..pos+S-1. Writes logits
// [S, vocab] to `logits`. The KV cache is [L, KV, max_len, hd]; hist is the
// full token history used by the engram.
static int forward(const needle_model_t* m, const int32_t* tokens, uint32_t S,
                   uint32_t pos, float* kv_cache, uint32_t cache_len,
                   const int32_t* hist, const float* cos_t, const float* sin_t,
                   float* logits) {
    const uint32_t L = m->num_layers;
    const uint32_t d = m->d_model;
    const uint32_t H = m->num_heads;
    const uint32_t KV = m->num_kv_heads;
    const uint32_t hd = m->head_dim;
    const uint32_t n = m->mhc_lanes;
    const uint32_t nC = n * d;
    const int NS = (int)S;
    const int reps = (int)(H / KV);

    // x: [S, n, d] lanes.
    float* x = (float*)xcalloc((size_t)S * n * d, sizeof(float));
    float* u = (float*)xcalloc((size_t)S * d, sizeof(float));
    float* nx = (float*)xcalloc((size_t)S * nC, sizeof(float));
    float* hpre = (float*)xcalloc((size_t)S * n, sizeof(float));
    float* hpost = (float*)xcalloc((size_t)S * n, sizeof(float));
    float* res = (float*)xcalloc((size_t)S * n * n, sizeof(float));
    float* hres = (float*)xcalloc((size_t)S * n * n, sizeof(float));
    float* h = (float*)xcalloc((size_t)S * d, sizeof(float));
    float* q = (float*)xcalloc((size_t)S * H * hd, sizeof(float));
    float* k = (float*)xcalloc((size_t)S * KV * hd, sizeof(float));
    float* v = (float*)xcalloc((size_t)S * KV * hd, sizeof(float));
    float* attn = (float*)xcalloc((size_t)S * d, sizeof(float));
    float* gate = (float*)xcalloc((size_t)S * d, sizeof(float));
    float* z = (float*)xcalloc((size_t)S * d, sizeof(float));
    float* y = (float*)xcalloc((size_t)S * d, sizeof(float));
    // Score scratch (bounded by the sliding window).
    float* scbuf = (float*)xcalloc((size_t)m->kv_window, sizeof(float));
    if (!x || !u || !nx || !hpre || !hpost || !res || !hres || !h || !q || !k ||
        !v || !attn || !gate || !z || !y || !scbuf) {
        goto oom;
    }

    // Embedding -> 4 lanes.
    {
        float scale = sqrtf((float)d);
        for (int s = 0; s < NS; s++) {
            const float* erow = m->embedding + (size_t)tokens[s] * d;
            for (uint32_t c = 0; c < d; c++) {
                float val = erow[c] * scale;
                for (uint32_t l = 0; l < n; l++) {
                    x[((size_t)s * n + l) * d + c] = val;
                }
            }
        }
    }

    // Engram for this block (positions pos..pos+S-1).
    float* ekv_k[2] = {NULL, NULL};
    float* ekv_v[2] = {NULL, NULL};
    for (uint32_t si = 0; si < m->num_engram_sites; si++) {
        ekv_k[si] = (float*)xcalloc((size_t)S * d, sizeof(float));
        ekv_v[si] = (float*)xcalloc((size_t)S * d, sizeof(float));
        if (!ekv_k[si] || !ekv_v[si]) goto oom;
        engram_kv(m, m->eg_tables[si], m->eg_key[si], m->eg_value[si], m->eg_taps[si],
                  hist, pos, S, ekv_k[si], ekv_v[si]);
    }

    for (uint32_t li = 0; li < L; li++) {
        // nx = rms_unit(flattened lanes).
        for (int s = 0; s < NS; s++) {
            float* xr = x + (size_t)s * n * d;
            float* nr = nx + (size_t)s * nC;
            float ss = 0.0f;
            for (uint32_t k2 = 0; k2 < nC; k2++) ss += xr[k2] * xr[k2];
            float rms = sqrtf(ss / (float)nC + SAN_EPS);
            for (uint32_t k2 = 0; k2 < nC; k2++) nr[k2] = xr[k2] / rms;
        }
        // hpre, then u = mix lanes.
        // Lane offsets are per-layer: layer i activates lane i % n, so
        // pre_off = 8*lane-4 = {4 if active else -4}, post_off = -4*(1-lane)
        // = {0 if active else -4}.
        uint32_t active = li % n;
        for (int s = 0; s < NS; s++) {
            const float* nr = nx + (size_t)s * nC;
            float* up = u + (size_t)s * d;
            memset(up, 0, d * sizeof(float));
            for (uint32_t l = 0; l < n; l++) {
                const float* phi = m->mhc_phi_pre + ((size_t)li * n + l) * nC;
                float dot = 0.0f;
                for (uint32_t k2 = 0; k2 < nC; k2++) dot += nr[k2] * phi[k2];
                float poff = (l == active) ? 4.0f : -4.0f;
                float hp = sigmoidf(m->mhc_a_pre[li] * dot + m->mhc_b_pre[(size_t)li * n + l] +
                                    poff);
                hpre[(size_t)s * n + l] = hp;
                const float* xl = x + ((size_t)s * n + l) * d;
                for (uint32_t c = 0; c < d; c++) up[c] += hp * xl[c];
            }
        }

        // Engram injection at the configured site layers.
        memcpy(y, u, (size_t)S * d * sizeof(float));
        if (m->num_engram_sites) {
            for (uint32_t si = 0; si < m->num_engram_sites; si++) {
                if (m->engram_sites[si] == li) {
                    for (int s = 0; s < NS; s++) {
                        const float* uu = u + (size_t)s * d;
                        const float* ekk = ekv_k[si] + (size_t)s * d;
                        const float* evv = ekv_v[si] + (size_t)s * d;
                        float dot = 0.0f, nu = 0.0f, ne = 0.0f;
                        for (uint32_t c = 0; c < d; c++) {
                            dot += uu[c] * ekk[c];
                            nu += uu[c] * uu[c];
                            ne += ekk[c] * ekk[c];
                        }
                        float rmsu = sqrtf(nu / (float)d + SAN_EPS);
                        float rmse = sqrtf(ne / (float)d + SAN_EPS);
                        float alpha = sigmoidf((dot / (rmsu * rmse)) / sqrtf((float)d));
                        float* yr = y + (size_t)s * d;
                        for (uint32_t c = 0; c < d; c++) yr[c] += alpha * evv[c];
                    }
                    break;
                }
            }
        }

        // Block: norm -> attention -> gate -> out_proj -> norm -> residual.
        memcpy(h, y, (size_t)S * d * sizeof(float));
        zcrms(h, m->norm_in + (size_t)li * d, NS, (int)d);

        // q/k/v projections.
        gemm_xwt(h, m->q_proj + (size_t)li * d * d, S, (int)d, (int)d, q);
        gemm_xwt(h, m->k_proj + (size_t)li * d * (KV * hd), S, (int)d, (int)(KV * hd), k);
        gemm_xwt(h, m->v_proj + (size_t)li * d * (KV * hd), S, (int)d, (int)(KV * hd), v);

        // reshape q -> [S, H, hd] (already row-major as [S, H*hd]); zcrms per head.
        zcrms_heads(q, m->q_norm + (size_t)li * hd, NS, (int)H, (int)hd);
        zcrms_heads(k, m->k_norm + (size_t)li * hd, NS, (int)KV, (int)hd);

        apply_rope(q, (int)H, k, (int)KV, S, (int)hd, (int)pos, cos_t, sin_t);

        // Write into the KV cache at positions pos..pos+S-1. The k and v
        // blocks are separate: [k_0..k_{L-1}][v_0..v_{L-1}].
        float* kc = kv_cache + (size_t)li * KV * cache_len * hd;
        float* vc = kv_cache + (size_t)(L * KV * cache_len + li * KV * cache_len) * hd;
        for (int s = 0; s < NS; s++) {
            for (uint32_t h = 0; h < KV; h++) {
                const float* kr = k + ((size_t)s * KV + h) * hd;
                const float* vr = v + ((size_t)s * KV + h) * hd;
                float* kdst = kc + ((size_t)h * cache_len + pos + s) * hd;
                float* vdst = vc + ((size_t)h * cache_len + pos + s) * hd;
                memcpy(kdst, kr, hd * sizeof(float));
                memcpy(vdst, vr, hd * sizeof(float));
            }
        }

        // Attention over the cache with causal + sliding-window mask.
        for (int s = 0; s < NS; s++) {
            int qpos = (int)pos + s;
            int kmin = qpos - (int)m->kv_window + 1;
            if (kmin < 0) kmin = 0;
            for (int qh = 0; qh < (int)H; qh++) {
                int kvh = qh / reps;
                const float* qr = q + ((size_t)s * H + qh) * hd;
                // Scores.
                int nk = qpos - kmin + 1;
                float* sc = scbuf;
                float mx = -1e30f;
                for (int kp = kmin; kp <= qpos; kp++) {
                    const float* kr = kc + ((size_t)kvh * cache_len + kp) * hd;
                    float acc = 0.0f;
                    for (uint32_t c = 0; c < hd; c++) acc += qr[c] * kr[c];
                    sc[kp - kmin] = acc / sqrtf((float)hd);
                    if (sc[kp - kmin] > mx) mx = sc[kp - kmin];
                }
                float sum = 0.0f;
                for (int i = 0; i < nk; i++) {
                    sc[i] = expf(sc[i] - mx);
                    sum += sc[i];
                }
                float* orow = attn + ((size_t)s * H + qh) * hd;
                for (uint32_t c = 0; c < hd; c++) {
                    float acc = 0.0f;
                    for (int kp = kmin; kp <= qpos; kp++) {
                        const float* vr2 = vc + ((size_t)kvh * cache_len + kp) * hd;
                        acc += sc[kp - kmin] / sum * vr2[c];
                    }
                    orow[c] = acc;
                }
            }
        }

        // gate: sigmoid(h @ gate_proj); out = attn * gate; out_proj.
        memcpy(gate, h, (size_t)S * d * sizeof(float));
        gemm_xwt(gate, m->gate_proj + (size_t)li * d * d, S, (int)d, (int)d, z);
        for (int s = 0; s < NS; s++) {
            float* gr = z + (size_t)s * d;
            float* ar = attn + (size_t)s * d;
            for (uint32_t c = 0; c < d; c++) ar[c] *= sigmoidf(gr[c]);
        }
        gemm_xwt(attn, m->out_proj + (size_t)li * d * d, S, (int)d, (int)d, z);
        memcpy(attn, z, (size_t)S * d * sizeof(float));
        zcrms(attn, m->post_norm + (size_t)li * d, NS, (int)d);

        // residual: x = bx + sigmoid(attn_gate) * attn; then hadamard MLP.
        float ag = sigmoidf(m->attn_gate[li]);
        for (int s = 0; s < NS; s++) {
            float* yr = y + (size_t)s * d;
            const float* ar = attn + (size_t)s * d;
            for (uint32_t c = 0; c < d; c++) yr[c] += ag * ar[c];
        }
        memcpy(h, y, (size_t)S * d * sizeof(float));
        zcrms(h, m->pre_hada + (size_t)li * d, NS, (int)d);
        // z = silu(d2 * (d1*h @ H)) @ H, scaled by d3, via fast Hadamard.
        {
            const float* d1w = m->d1 + (size_t)li * d;
            const float* d2w = m->d2 + (size_t)li * d;
            const float* d3w = m->d3 + (size_t)li * d;
            float inv = 1.0f / sqrtf((float)m->hada_n);
            int hn = (int)m->hada_n;
            for (int s = 0; s < NS; s++) {
                float* hr = h + (size_t)s * d;
                float* zr = z + (size_t)s * d;
                // Hadamard MLP on `hr`, output into zr (d == hada_n here).
                for (uint32_t c = 0; c < d; c++) zr[c] = d1w[c] * hr[c];
                fwht(zr, hn);
                for (uint32_t c = 0; c < d; c++) zr[c] *= inv;
                for (uint32_t c = 0; c < d; c++) zr[c] = siluf(d2w[c] * zr[c]);
                fwht(zr, hn);
                for (uint32_t c = 0; c < d; c++) zr[c] *= inv;
                for (uint32_t c = 0; c < d; c++) zr[c] *= d3w[c];
            }
        }
        for (int s = 0; s < NS; s++) {
            float* yr = y + (size_t)s * d;
            const float* zr = z + (size_t)s * d;
            for (uint32_t c = 0; c < d; c++) yr[c] += zr[c];
        }

        // MHC update: hpost, hres (Sinkhorn), new lanes.
        for (int s = 0; s < NS; s++) {
            const float* nr = nx + (size_t)s * nC;
            float* hpr = hpost + (size_t)s * n;
            for (uint32_t l = 0; l < n; l++) {
                const float* phi = m->mhc_phi_post + ((size_t)li * n + l) * nC;
                float dot = 0.0f;
                for (uint32_t k2 = 0; k2 < nC; k2++) dot += nr[k2] * phi[k2];
                float soff = (l == active) ? 0.0f : -4.0f;
                hpr[l] = 2.0f * sigmoidf(m->mhc_a_post[li] * dot +
                                         m->mhc_b_post[(size_t)li * n + l] + soff);
            }
            for (uint32_t r = 0; r < n * n; r++) {
                const float* phi = m->mhc_phi_res + ((size_t)li * n * n + r) * nC;
                float dot = 0.0f;
                for (uint32_t k2 = 0; k2 < nC; k2++) dot += nr[k2] * phi[k2];
                res[(size_t)s * n * n + r] = dot;
            }
            float hres_in[16];
            for (uint32_t l1 = 0; l1 < n; l1++) {
                for (uint32_t l2 = 0; l2 < n; l2++) {
                    hres_in[l1 * n + l2] =
                        m->mhc_a_res[li] * res[(size_t)s * n * n + l1 * n + l2] +
                        m->mhc_b_res[(size_t)li * n * n + l1 * n + l2];
                }
            }
            sinkhorn44(hres_in, hres + (size_t)s * n * n);
        }
        {
            float* newx = (float*)xcalloc((size_t)S * n * d, sizeof(float));
            if (!newx) goto oom;
            for (int s = 0; s < NS; s++) {
                const float* hpr = hpost + (size_t)s * n;
                const float* hr16 = hres + (size_t)s * n * n;
                const float* oldx = x + (size_t)s * n * d;
                float* nwx = newx + (size_t)s * n * d;
                const float* blk = y + (size_t)s * d;
                const float* uu = u + (size_t)s * d;
                for (uint32_t l = 0; l < n; l++) {
                    for (uint32_t c = 0; c < d; c++) {
                        float acc = 0.0f;
                        for (uint32_t l2 = 0; l2 < n; l2++) {
                            acc += hr16[l * n + l2] * oldx[(size_t)l2 * d + c];
                        }
                        nwx[(size_t)l * d + c] = acc + hpr[l] * (blk[c] - uu[c]);
                    }
                }
            }
            memcpy(x, newx, (size_t)S * n * d * sizeof(float));
            free(newx);
        }
    }

    // Final: mean over lanes, final norm, tied logits.
    for (int s = 0; s < NS; s++) {
        const float* xr = x + (size_t)s * n * d;
        float* hr = h + (size_t)s * d;
        for (uint32_t c = 0; c < d; c++) {
            float acc = 0.0f;
            for (uint32_t l = 0; l < n; l++) acc += xr[(size_t)l * d + c];
            hr[c] = acc / (float)n;
        }
    }
    zcrms(h, m->final_norm, NS, (int)d);
    gemm_xwt(h, m->embedding, S, (int)d, (int)m->vocab, logits);

    for (int si = 0; si < 2; si++) {
        free(ekv_k[si]);
        free(ekv_v[si]);
    }
    free(x);
    free(u);
    free(nx);
    free(hpre);
    free(hpost);
    free(res);
    free(hres);
    free(h);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(gate);
    free(z);
    free(y);
    free(scbuf);
    return 0;

oom:
    for (int si = 0; si < 2; si++) {
        free(ekv_k[si]);
        free(ekv_v[si]);
    }
    free(x);
    free(u);
    free(nx);
    free(hpre);
    free(hpost);
    free(res);
    free(hres);
    free(h);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(gate);
    free(z);
    free(y);
    free(scbuf);
    return -1;
}

// --- generate -------------------------------------------------------------

static int rope_tables(const needle_model_t* m, uint32_t max_len, float** cos_out,
                       float** sin_out) {
    int half = (int)m->head_dim / 2;
    float* c = (float*)malloc((size_t)max_len * half * sizeof(float));
    float* sn = (float*)malloc((size_t)max_len * half * sizeof(float));
    if (!c || !sn) {
        free(c);
        free(sn);
        return -1;
    }
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(m->rope_theta, 2.0f * (float)i / (float)m->head_dim);
        for (uint32_t t = 0; t < max_len; t++) {
            float ang = (float)t * freq;
            c[(size_t)t * half + i] = cosf(ang);
            sn[(size_t)t * half + i] = sinf(ang);
        }
    }
    *cos_out = c;
    *sin_out = sn;
    return 0;
}

uint32_t needle_model_vocab(const needle_model_t* m) { return m->vocab; }

int needle_model_prompt_logits(const needle_model_t* m, const needle_tokenizer_t* tok,
                               const char* prompt, float* logits, size_t logits_cap) {
    if (!m || !tok || !logits || logits_cap < m->vocab) return -1;
    int prompt_ids[4096];
    int n_prompt = needle_tokenizer_encode(tok, prompt, prompt_ids, 4096);
    if (n_prompt < 0 || n_prompt + 1 >= 4096) return -1;
    size_t plen = (size_t)n_prompt + 1;
    uint32_t max_len = m->max_seq_len;
    if (plen < max_len) max_len = (uint32_t)plen;

    float* cos_t = NULL;
    float* sin_t = NULL;
    if (rope_tables(m, max_len, &cos_t, &sin_t) != 0) return -1;
    // Two caches (k and v) per layer: [L, KV, max_len, hd] each.
    size_t kv_n = (size_t)m->num_layers * m->num_kv_heads * max_len * m->head_dim;
    float* kv_cache = (float*)xcalloc(kv_n ? 2 * kv_n : 1, sizeof(float));
    int32_t* hist = (int32_t*)xcalloc(max_len, sizeof(int32_t));
    float* full = (float*)xcalloc((size_t)max_len * m->vocab, sizeof(float));
    int32_t* ptoks = (int32_t*)malloc(plen * sizeof(int32_t));
    if (!kv_cache || !hist || !full || !ptoks) {
        free(cos_t);
        free(sin_t);
        free(kv_cache);
        free(hist);
        free(full);
        free(ptoks);
        return -1;
    }
    ptoks[0] = (int32_t)tok->bos_id;
    for (int i = 0; i < n_prompt; i++) ptoks[1 + i] = prompt_ids[i];
    for (size_t i = 0; i < plen; i++) hist[i] = ptoks[i];

    int rc = forward(m, ptoks, (uint32_t)plen, 0, kv_cache, max_len, hist, cos_t, sin_t, full);
    if (rc == 0) {
        memcpy(logits, full + (plen - 1) * m->vocab, m->vocab * sizeof(float));
    }
    free(cos_t);
    free(sin_t);
    free(kv_cache);
    free(hist);
    free(full);
    free(ptoks);
    return rc == 0 ? (int)plen : -1;
}

static int32_t* run_generate(const needle_model_t* m, const needle_tokenizer_t* tok,
                             const char* prompt, int max_new_tokens, float temperature,
                             size_t* n_out) {
    int bos = (int)tok->bos_id;
    int eos = (int)tok->eos_id;

    int prompt_ids[4096];
    int n_prompt = needle_tokenizer_encode(tok, prompt, prompt_ids, 4096);
    if (n_prompt < 0) return NULL;
    if (n_prompt + 1 >= 4096) return NULL;

    uint32_t max_len = m->max_seq_len;
    uint32_t need = (uint32_t)(n_prompt + 1 + (max_new_tokens > 0 ? max_new_tokens : 0));
    if (need < max_len) max_len = need;

    float* cos_t = NULL;
    float* sin_t = NULL;
    if (rope_tables(m, max_len, &cos_t, &sin_t) != 0) return NULL;

    // KV cache: [L, KV, max_len, hd].
    // Two caches (k and v) per layer: [L, KV, max_len, hd] each.
    size_t kv_n = (size_t)m->num_layers * m->num_kv_heads * max_len * m->head_dim;
    float* kv_cache = (float*)xcalloc(kv_n ? 2 * kv_n : 1, sizeof(float));
    int32_t* hist = (int32_t*)xcalloc(max_len, sizeof(int32_t));
    int32_t* gen_ids = (int32_t*)xcalloc((size_t)(max_new_tokens > 0 ? max_new_tokens : 1),
                                         sizeof(int32_t));
    float* logits = (float*)xcalloc((size_t)m->vocab, sizeof(float));
    // Prompt logits need [plen * vocab]; read the last row into `logits`.
    float* full = (float*)xcalloc((size_t)max_len * m->vocab, sizeof(float));
    if (!kv_cache || !hist || !gen_ids || !logits || !full) {
        free(cos_t);
        free(sin_t);
        free(kv_cache);
        free(hist);
        free(gen_ids);
        free(logits);
        free(full);
        return NULL;
    }

    // Prompt buffer: [BOS] + ids.
    size_t plen = (size_t)n_prompt + 1;
    int32_t* ptoks = (int32_t*)malloc(plen * sizeof(int32_t));
    if (!ptoks) {
        free(cos_t);
        free(sin_t);
        free(kv_cache);
        free(hist);
        free(gen_ids);
        free(logits);
        return NULL;
    }
    ptoks[0] = bos;
    for (int i = 0; i < n_prompt; i++) ptoks[1 + i] = prompt_ids[i];
    for (size_t i = 0; i < plen; i++) hist[i] = ptoks[i];

    if (forward(m, ptoks, (uint32_t)plen, 0, kv_cache, max_len, hist, cos_t, sin_t, full) != 0) {
        free(cos_t);
        free(sin_t);
        free(kv_cache);
        free(hist);
        free(gen_ids);
        free(logits);
        free(full);
        free(ptoks);
        return NULL;
    }
    memcpy(logits, full + (plen - 1) * m->vocab, m->vocab * sizeof(float));
    // Greedy next from the last prompt position (logits = last row).
    int nxt = 0;
    {
        const float* last = logits;
        float best = last[0];
        for (uint32_t i = 1; i < m->vocab; i++) {
            if (last[i] > best) {
                best = last[i];
                nxt = (int)i;
            }
        }
    }

    size_t n_gen = 0;
    uint32_t pos = (uint32_t)plen;
    while (n_gen < (size_t)max_new_tokens && pos < max_len) {
        if (nxt == eos) break;
        gen_ids[n_gen++] = nxt;
        hist[pos] = nxt;
        int32_t one = nxt;
        if (forward(m, &one, 1, pos, kv_cache, max_len, hist, cos_t, sin_t, logits) != 0) {
            break;
        }
        if (temperature <= 0.0f) {
            nxt = 0;
            float best = logits[0];
            for (uint32_t i = 1; i < m->vocab; i++) {
                if (logits[i] > best) {
                    best = logits[i];
                    nxt = (int)i;
                }
            }
        } else {
            // Temperature sampling with a simple LCG PRNG.
            static unsigned long long rng_state = 0x123456789ABCDEFull;
            rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
            double u = (double)(rng_state >> 11) / (1ULL << 53);
            double t = (double)temperature;
            double sum = 0.0;
            int picked = (int)m->vocab - 1;
            for (int i = 0; i < (int)m->vocab; i++) {
                sum += exp((double)logits[i] / t);
            }
            double target = u * sum;
            double acc = 0.0;
            for (int i = 0; i < (int)m->vocab; i++) {
                acc += exp((double)logits[i] / t);
                if (acc >= target) {
                    picked = i;
                    break;
                }
            }
            nxt = picked;
        }
        pos++;
    }

    if (n_out) *n_out = n_gen;
    free(cos_t);
    free(sin_t);
    free(kv_cache);
    free(hist);
    free(logits);
    free(full);
    free(ptoks);
    return gen_ids;
}

int32_t* needle_model_generate_ids(const needle_model_t* m, const needle_tokenizer_t* tok,
                                   const char* prompt, int max_new_tokens, float temperature,
                                   size_t* n_out) {
    size_t n = 0;
    int32_t* ids = run_generate(m, tok, prompt, max_new_tokens, temperature, &n);
    if (n_out) *n_out = n;
    return ids;
}

char* needle_model_generate(const needle_model_t* m, const needle_tokenizer_t* tok,
                            const char* prompt, int max_new_tokens, float temperature,
                            FILE* stream) {
    size_t n_gen = 0;
    int32_t* gen_ids = run_generate(m, tok, prompt, max_new_tokens, temperature, &n_gen);
    if (!gen_ids) return NULL;

    char* prev_dec = NULL;
    if (stream) {
        for (size_t i = 1; i <= n_gen; i++) {
            char* dec = needle_tokenizer_decode(tok, gen_ids, i);
            if (!dec) continue;
            if (prev_dec) {
                size_t pl = strlen(prev_dec);
                if (strncmp(prev_dec, dec, pl) == 0) {
                    fputs(dec + pl, stream);
                } else {
                    fputs(dec, stream);
                }
            } else {
                fputs(dec, stream);
            }
            fflush(stream);
            free(prev_dec);
            prev_dec = dec;
        }
    }

    char* out = needle_tokenizer_decode(tok, gen_ids, n_gen);
    free(prev_dec);
    free(gen_ids);
    return out;
}
