# boat 框架后续方向建议

基于最近的工作（CUDA 后端、NanoChat LLM、Qwen3-ASR、Nougat-LaTeX OCR、LLM Serving API），以下是几个值得考虑的后续方向：

## 1. LLM Serving 集成 OCR 模型

已有的 OpenAI 兼容 HTTP 服务（来自 NanoChat）可以扩展支持 OCR 模型。将 Nougat-LaTeX 或 GLM-OCR 封装为 API 端点，接收图片返回 LaTeX 公式，方便与其他工具集成。

**关键文件：** `examples/llm_server/`，`examples/latex/`

## 2. 通用视觉编码器（ViT）

当前 Swin 实现是特定于 Nougat 的。构建一个通用 ViT 编码器可以解锁更多模型：CLIP、DINOv2、SigLIP 等。框架已有必要的基础组件（Multi-Head Attention、LayerNorm、GELU），只需组合成标准 ViT 架构。

**新文件：** `include/boat/layers/vit.h`，`src/layers/vit.c`

## 3. VLM（视觉语言模型）

将视觉编码器（Swin/ViT）与 NanoChat LLM 解码器通过投影层连接，类似 LLaVA 架构。vision 和 language 能力已在框架中各自就绪，组合是自然延伸。

## 4. 模型导出管道（Python 工具）

为 boat 编写一个 Python 转换工具，将任意 HuggingFace 模型（PyTorch `.bin` / `.safetensors`）转换为 boat 原生格式。当前每个模型都需要手写 C 加载器——统一的格式可以大幅降低移植新模型的成本。

**新文件：** `scripts/export_to_boat.py`

## 5. Qwen3-ASR 改进

ASR 流水线已可运行，可以继续增强：
- 非贪心解码（beam search）
- 标点恢复（punctuation restoration）
- 语音活动检测（VAD），支持实时处理

**相关文件：** `examples/qwen3_asr/`，`src/cuda/`

## 6. 量化工具包

框架已支持 BITS1/BITS2/BITS4/BITS8 数据类型。开发一个量化工具，使用校准数据将浮点模型量化为低比特格式，可以显著降低大模型的内存占用和推理延迟。

**新文件：** `scripts/quantize.py`，`src/quantization/`
