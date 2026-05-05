// config.h — Qwen3-ASR-0.6B model hyperparameters
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef QWEN3ASR_CONFIG_H
#define QWEN3ASR_CONFIG_H

// Audio encoder
#define QWEN3ASR_ENCODER_D_MODEL      896
#define QWEN3ASR_ENCODER_NUM_HEADS    14
#define QWEN3ASR_ENCODER_HEAD_DIM     64
#define QWEN3ASR_ENCODER_FFN_DIM      3584
#define QWEN3ASR_ENCODER_NUM_LAYERS   18
#define QWEN3ASR_ENCODER_OUTPUT_DIM   1024
#define QWEN3ASR_ENCODER_MAX_SRC_POS  1500

// Conv frontend
#define QWEN3ASR_CONV_DOWNSAMPLE      480
#define QWEN3ASR_MEL_BINS             128
#define QWEN3ASR_CONV_CHUNK_SIZE      100   // process mel in 100-frame chunks

// Text decoder
#define QWEN3ASR_DECODER_HIDDEN_SIZE    1024
#define QWEN3ASR_DECODER_INTERMEDIATE   3072
#define QWEN3ASR_DECODER_NUM_HEADS      16
#define QWEN3ASR_DECODER_NUM_KV_HEADS   8
#define QWEN3ASR_DECODER_HEAD_DIM       128
#define QWEN3ASR_DECODER_NUM_LAYERS     28
#define QWEN3ASR_VOCAB_SIZE             151936
#define QWEN3ASR_RMS_EPS                1e-6f
#define QWEN3ASR_ROPE_THETA             1000000.0f
#define QWEN3ASR_MAX_SEQ_LEN            8192
#define QWEN3ASR_MAX_NEW_TOKENS         256

// Special token IDs (Qwen2Tokenizer convention)
#define QWEN3ASR_AUDIO_PLACEHOLDER_ID   151676
#define QWEN3ASR_IM_START_ID            151644
#define QWEN3ASR_IM_END_ID              151645
#define QWEN3ASR_EOS_ID                 151645  // im_end is used as EOS
#define QWEN3ASR_PAD_ID                 0

#endif // QWEN3ASR_CONFIG_H
