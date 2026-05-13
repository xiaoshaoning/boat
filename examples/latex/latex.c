// latex.c - Nougat-LaTeX OCR main entry point
// Usage: latex_ocr <model_dir> <image_path> [--cpu]
//
// Loads a Nougat-LaTeX model, runs OCR on the input image,
// and prints the predicted LaTeX markup to stdout.
#include "nougat_model.h"
#include "nougat_decoder.h"
#include "image.h"
#include <boat/tensor.h>
#include <boat/layers/swin.h>
#include <boat/tokenizers/bpe.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len <= 0) { fclose(f); return NULL; }
    rewind(f);
    char* buf = (char*)malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf); fclose(f); return NULL;
    }
    buf[len] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)len;
    return buf;
}

static int file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <image_path> [--cpu]\n", argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* image_path = argv[2];
    boat_device_t device = BOAT_DEVICE_CUDA;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) {
            device = BOAT_DEVICE_CPU;
        }
    }

    // Verify paths
    char model_path[1024];
    snprintf(model_path, sizeof(model_path), "%s/model.safetensors", model_dir);
    if (!file_exists(model_path)) {
        fprintf(stderr, "Error: model not found at %s\n", model_path);
        return 1;
    }
    if (!file_exists(image_path)) {
        fprintf(stderr, "Error: image not found at %s\n", image_path);
        return 1;
    }
    char tok_path[1024];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    if (!file_exists(tok_path)) {
        fprintf(stderr, "Error: tokenizer not found at %s\n", tok_path);
        return 1;
    }

    printf("[Nougat] Loading model from %s ...\n", model_dir);
    fflush(stdout);

    // ---- Load model ----
    nougat_model_t* model = nougat_model_create(model_dir);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }
    printf("[Nougat] Model loaded (%d decoder layers)\n", model->num_decoder_layers);

    // Move to device
    if (device == BOAT_DEVICE_CUDA) {
        printf("[Nougat] Moving model to GPU ...\n");
        fflush(stdout);
        if (!nougat_model_to_device(model, device)) {
            fprintf(stderr, "Error: failed to move model to GPU\n");
            nougat_model_free(model);
            return 1;
        }
    }

    // ---- Load tokenizer ----
    printf("[Nougat] Loading tokenizer from %s ...\n", tok_path);
    fflush(stdout);
    boat_bpe_tokenizer_t* tokenizer = boat_bpe_tokenizer_create(tok_path);
    if (!tokenizer) {
        fprintf(stderr, "Error: failed to load tokenizer\n");
        nougat_model_free(model);
        return 1;
    }

    // ---- Load image ----
    printf("[Nougat] Loading image %s ...\n", image_path);
    fflush(stdout);
    int img_w, img_h;
    uint8_t* pixels = nougat_load_image(image_path, &img_w, &img_h);
    if (!pixels) {
        fprintf(stderr, "Error: failed to load image (BMP/PPM only)\n");
        boat_bpe_tokenizer_free(tokenizer);
        nougat_model_free(model);
        return 1;
    }
    printf("[Nougat] Image loaded: %dx%d\n", img_w, img_h);

    // Convert to tensor [1, 3, H, W] on CPU then move to device
    boat_tensor_t* img_tensor = nougat_image_to_tensor(pixels, img_h, img_w, BOAT_DEVICE_CPU);
    free(pixels);
    if (!img_tensor) {
        fprintf(stderr, "Error: failed to convert image to tensor\n");
        boat_bpe_tokenizer_free(tokenizer);
        nougat_model_free(model);
        return 1;
    }

    if (device == BOAT_DEVICE_CUDA) {
        boat_tensor_t* d_img = boat_tensor_to_device(img_tensor, device);
        boat_tensor_unref(img_tensor);
        img_tensor = d_img;
        if (!img_tensor) {
            fprintf(stderr, "Error: failed to move image tensor to GPU\n");
            boat_bpe_tokenizer_free(tokenizer);
            nougat_model_free(model);
            return 1;
        }
    }

    // ---- Run encoder (Swin Transformer) ----
    printf("[Nougat] Running encoder (Swin) ...\n");
    fflush(stdout);
    boat_tensor_t* encoder_output = boat_swin_forward(
        &model->swin_config, model->encoder, img_tensor);
    boat_tensor_unref(img_tensor);
    if (!encoder_output) {
        fprintf(stderr, "Error: encoder forward failed\n");
        boat_bpe_tokenizer_free(tokenizer);
        nougat_model_free(model);
        return 1;
    }

    const int64_t* eshape = boat_tensor_shape(encoder_output);
    printf("[Nougat] Encoder output: [%lld, %lld, %lld]\n",
           (long long)eshape[0], (long long)eshape[1], (long long)eshape[2]);

    // ---- Run decoder (autoregressive generation) ----
    printf("[Nougat] Generating LaTeX ...\n");
    fflush(stdout);

    int max_steps = 1024;
    int32_t* out_ids = NULL;
    int out_len = 0;

    int ret = nougat_decoder_generate(
        model, encoder_output, tokenizer, max_steps, device, &out_ids, &out_len);
    boat_tensor_unref(encoder_output);

    if (ret != 0 || !out_ids) {
        fprintf(stderr, "Error: decoder generation failed\n");
        boat_bpe_tokenizer_free(tokenizer);
        nougat_model_free(model);
        return 1;
    }

    printf("[Nougat] Generated %d tokens\n", out_len);

    // ---- Decode tokens to text ----
    char* latex_output = boat_bpe_tokenizer_decode(tokenizer, out_ids, (size_t)out_len);
    free(out_ids);

    if (!latex_output) {
        fprintf(stderr, "Error: tokenizer decode failed\n");
        boat_bpe_tokenizer_free(tokenizer);
        nougat_model_free(model);
        return 1;
    }

    // ---- Print result ----
    printf("\n========== LaTeX Output ==========\n");
    printf("%s", latex_output);
    printf("\n==================================\n");

    // ---- Cleanup ----
    free(latex_output);
    boat_bpe_tokenizer_free(tokenizer);
    nougat_model_free(model);

    printf("[Nougat] Done.\n");
    return 0;
}
