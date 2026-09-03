<div align="center">

# Intern Lumina U2：面向全视觉模态理解、图像生成与编辑的多码本扩散大语言模型

*上海人工智能实验室*

[[📑 技术报告（即将发布）](#)] &emsp; [[🌐 项目主页](https://internlm.github.io/InternLumina-U2/)] &emsp; [[🤗 模型](https://huggingface.co/internlm/InternLumina-U2)]

[English](./README.md) | [简体中文](./README_zh-CN.md)

</div>

![teaser](assets/teaser.png)

## 📰 新闻

- **[2026-08]** 🚀 发布推理代码（本仓库）。模型权重、训练代码与技术报告即将发布 — 见[规划](#%EF%B8%8F-规划)。

## 🌟 简介

Intern Lumina U2 是一个统一多模态模型，将语言、图像、视频与 3D 纳入同一框架，以一个模型覆盖**文本问答、文生图、图像理解、图像编辑、视频理解与 3D 理解**。模型为 **16B 总参数、1B 激活参数（16B-A1B）的 MoE**，以高效稀疏骨干搭配基于 [AToken](https://github.com/apple/ml-atoken) 的 **8 码本全离散视觉表示**。

MMaDA、Lumina-DiMOO 等离散统一模型表明：语言与视觉一旦共享离散 token 接口，理解与生成便不再需要各自独立的模型体系。但从"能够统一"到"高性能统一"，瓶颈在于视觉内容如何被表示和预测：单码本限制了单个视觉 token 的信息容量，而离散—连续混合路线又把理解与生成拆到不同的表示与解码路径上。

Intern Lumina U2 探索了一条新路径：**多码本全离散建模**。

![overview](assets/img1.jpg)

## ✨ 亮点

**1. 一个模型覆盖全视觉模态理解、图像生成与编辑。** 文本问答、文生图、图像理解（VQA / OCR / 图表 / 数学推理）、指令式图像编辑、视频理解与 3D 理解全部运行在同一骨干与同一离散 token 接口之上。

![capabilities](assets/img2.png)

**2. 多码本视觉表示。** [AToken](https://github.com/apple/ml-atoken) 分词器用 **8 个互补码本**共同描述每个视觉位置，在不增加序列长度的前提下扩展编码空间，保留局部纹理、文字、几何结构与高频细节。**输入侧**，每个码本保留独立 embedding，8 路逐位置 embedding 拼接后投影为单个骨干 token；**输出侧**采用*空间并行、码本深度自回归*：dLLM 骨干并行去噪多个被掩码的空间位置，多码本 AR head 按码本顺序预测每个位置的 8 个离散码。

![input](assets/img3.png)
<p align="center"><em>(a) 输入侧设计：各码本 embedding 拼接并投影为单个骨干 token。</em></p>

![output](assets/img4.png)
<p align="center"><em>(b) 输出侧设计：空间并行去噪 + 码本深度自回归 head。</em></p>

**3. 基于 LLaDA-2.0 MoE 的架构。** 语言骨干为 LLaDA-2.0 MoE 扩散语言模型（16B-A1B），其块扩散目标、稀疏专家容量与语言能力通过联合多任务训练扩展至统一视觉任务。

**4. 双硬件生态。** Intern Lumina U2 全面适配 **NVIDIA GPU 与昇腾 NPU**，覆盖训练、评测与推理，完成双后端算子级数值对齐并构建可平滑切换的端到端管线。在昇腾侧，通过 FLA、GMM 等融合算子缓解 MoE 稀疏架构与长序列训练的访存与 kernel launch 瓶颈、HSDP 分片寻优平衡通信与显存、gradient checkpointing 提升单卡最大序列长度等系统性优化，千卡级昇腾集群训练效率提升 **1.85 倍**。

## 🏆 评测结果

| 类别 | 基准 | 分数 |
|---|---|---|
| 图表 / 文档理解 | ChartQA | 86.52 |
| | CharXiv-DQ | 83.65 |
| 视觉数学推理 | MathVision | 33.22 |
| | DynaMath | 56.37 |
| 幻觉鲁棒性 | HallusionBench | 62.15 |
| 视频理解 | VideoMME | 51.26 |
| | MVBench | 59.74 |
| 3D 理解 | 3D MM-Vet | 41.8 |
| 文生图 | TIIF-Bench (Short/Long) | 84.60/85.95 |
| | DPG-Bench | 87.10 |
| 图像编辑 | ImgEdit | 3.83 |

在细粒度理解与生成基准上，Intern Lumina U2 超越 InternVL-U、LLaDA2.0-Uni、Show-o2、Lumina-DiMOO 等统一模型，并在 MathVision 与 DynaMath 上领先理解专用模型 InternVL3-8B。完整对比表见后续技术报告。

## 📦 发布范围

> **本仓库仅包含推理代码。** 模型权重、训练代码（独立仓库）、昇腾适配与技术报告将陆续发布 — 见[规划](#%EF%B8%8F-规划)。

## 🛠️ 安装

测试环境：**Python 3.11、PyTorch 2.6.0 + CUDA 12.4、flash-attn 2.7.2**。LLM 加载器将骨干固定在 PyTorch SDPA 上，但 AToken 视觉分词器需要 FlashAttention。

```bash
# 在本地仓库目录中：
cd InternLumina-U2

# 0) 创建环境
conda create -n intern-lumina-u2 python=3.11 -y
conda activate intern-lumina-u2

# 1) PyTorch（CUDA 12.4 版本）— 先于其余依赖安装
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2) 项目依赖（锁定为我们的测试版本）
pip install -r requirements.txt

# 3) flash-attn — AToken 视觉分词器需要（需 CUDA 工具链，基于上面的 torch 构建）
pip install flash-attn==2.7.2.post1 --no-build-isolation

# 4) VeOmni（vendor 自上游 @600fe6d，附本模型所需的本地兼容补丁）。
#    使用 --no-deps：依赖以 requirements.txt 为准。
pip install -e VeOmni/ --no-deps
```

说明：

- 导入模型代码需要可见 GPU（注意力 / MoE 内核在导入时初始化），请在 GPU 机器上运行。
- **不需要** `liger-kernel` 与 `xformers`。LLM 骨干运行在 PyTorch SDPA 上；`flash-attn` 仅 AToken 视觉分词器需要（即所有涉及图像 / 视频 / 3D 编解码的任务 — 纯文本生成无需安装）。
- 对象存储包（`petrel-oss-sdk` / `boto3`）为**可选** — 本地 token 文件无需它们。
- 预计算的 `.pkl` / `.pt` token 文件通过 PyTorch 可信负载模式反序列化（以兼容旧的 NumPy 负载），请仅加载可信来源的文件。自行生成请见[分词工具](#-分词工具)。
- 根驱动脚本刻意保持单进程（`INFER_NPROC_PER_NODE=1`），每个进程处理一个请求，不会将 batch 切分到多卡。
- 根驱动脚本会先切换到仓库根目录再启动 Python，因此相对路径的 checkpoint / 素材 / token / 输出均相对该根目录解析。在其他目录调用时请设置 `LUMINA_ROOT=/path/to/InternLumina-U2`；直接调用 Python 入口时请使用绝对路径或在仓库根目录运行。
- `requirements.txt` 中安装 `trimesh`，因为内置的 GLB/GLTF 编码器与网格回退路径需要它；`plyfile` 与 `diff-gaussian-rasterization` 是面向外部高斯泼溅资产的可选扩展。

## 🖥️ 昇腾 NPU

昇腾 NPU 的适配位于 [`ascend-npu-support`](https://github.com/InternLM/InternLumina-U2/tree/ascend-npu-support) 分支，而非 `main`。该分支的 LLM 骨干运行在 `torch_npu` 融合注意力内核上，融合 MoE 经由 NPU grouped matmul，因此需要 CANN 与配套的 `torch_npu`，取代上面的 CUDA 工具链与 `flash-attn`。

## 🎨 推理

所有推理经由同一个驱动脚本（`infer_1024_sft.sh`），每个任务一个 section。它需要一个 **split** 格式的 HF checkpoint 目录（`.../hf_ckpt_split`；权重即将发布，见规划），以及 — 凡涉及图像编解码 — 位于 `atoken_inference/checkpoints/atoken-sod.pt` 的 AToken 分词器权重（或经 `ATOKEN_MODEL_PATH=...` 指定）。使用 Transformers `AutoModel*` 时，模型资产目录须同时包含 `config.json`、`configuration_internluminau2.py`、`modeling_internluminau2.py` 与 `tokenbridge.py`。System prompt 属于 checkpoint 的训练契约：如需严格复现，请用训练该 checkpoint 时的原始 prompt 覆盖对应的 `*_SYSTEM_PROMPT` 变量。已有输出默认会被覆盖；仅当整个请求完全不变时才向 Python 入口传 `--skip_existing`。

底层 Python 入口（`infer_t2i_sft.py`、`infer_mmu_sft.py`、`infer_video_sft.py`、`infer_3d_sft.py`）也可直接调用，任意入口加 `--help` 查看参数。模型底层的 `generate` 是块扩散 API，请使用这些任务入口，而非标准 Transformers 的 `generate(max_length=...)` 接口。

### 1. 文本生成

```bash
SECTION=text TEXT_PROMPT="Explain why the sky appears blue in a few sentences." \
  CHECKPOINT=/path/to/hf_ckpt_split bash infer_1024_sft.sh
```

### 2. 文生图

`T2I_SIZE`（HxW，默认 1024x1024）、`T2I_CFG`（3.0）、`T2I_TEMP`（1.0）、`T2I_TIMESTEPS`（96）。

```bash
SECTION=t2i T2I_PROMPT="A serene mountain landscape with towering snow-capped peaks, a crystal-clear blue lake reflecting the mountains, dense pine forests, and a vibrant orange sunrise illuminating the sky." \
  T2I_SIZE=1024x1024 T2I_CFG=3.0 T2I_TIMESTEPS=96 \
  CHECKPOINT=/path/to/hf_ckpt_split bash infer_1024_sft.sh
```

### 3. 图像理解

`UNDER_MAX_EDGE` 限制编码分辨率（默认 512）；不设 `UNDER_PROMPT` 即为描述图片；不设 `UNDER_IMAGE` 则运行仓库自带的示例图。解码参数：`UNDER_GEN_LEN`（4096）、`UNDER_BLOCK_LEN`（32）、`UNDER_TIMESTEPS`（32）。

自然图像（VQA / 描述）：

```bash
SECTION=under UNDER_IMAGE=imgs/natural.jpg \
  UNDER_PROMPT="Could you provide a caption that neatly summarizes the image without losing focus on the important visuals?" \
  UNDER_MAX_EDGE=512 UNDER_GEN_LEN=4096 UNDER_TIMESTEPS=32 \
  CHECKPOINT=... bash infer_1024_sft.sh
```

图表 / 表格 / 文档问答（同一入口，信息密度更高的输入）：

```bash
SECTION=under UNDER_IMAGE=imgs/graph.png \
  UNDER_PROMPT="Is this a chart or a table? Reproduce it in Markdown and summarize what it shows." \
  CHECKPOINT=... bash infer_1024_sft.sh
```

### 4. 图像编辑

原始源图会被现场编码；默认运行仓库自带示例（`imgs/edit_source.png` + 炭笔画指令）。`EDIT_MAX_EDGE` 限制编码分辨率；`EDIT_CFG`（1）、`EDIT_TIMESTEPS`（64）。

```bash
SECTION=edit EDIT_IMAGE=imgs/edit_source.png \
  EDIT_INSTRUCTION="Convert the scene to a charcoal sketch style." \
  EDIT_MAX_EDGE=512 EDIT_TIMESTEPS=64 EDIT_CFG=1 \
  CHECKPOINT=... bash infer_1024_sft.sh
```

如需双 CFG，请同时设置两个变量（并省略 `EDIT_CFG`）：`EDIT_CFG_TEXT=0.5 EDIT_CFG_IMG=0.5`。

### 5. 视频理解

默认运行仓库自带示例视频（`imgs/demo_video.mp4`）；`VIDEO_PROMPT` 指定问题。`VIDEO_NUM_FRAMES`（采样 64 帧）、`VIDEO_MAX_SIZE`（每帧长边 448）、`VIDEO_GEN_LEN`（1024）、`VIDEO_BLOCK_LEN`（32）、`VIDEO_TIMESTEPS`（32）。

```bash
SECTION=video VIDEO_PATH=imgs/demo_video.mp4 \
  VIDEO_PROMPT="Describe this video in detail from start to finish." \
  VIDEO_NUM_FRAMES=64 VIDEO_MAX_SIZE=448 \
  CHECKPOINT=... bash infer_1024_sft.sh
```

### 6. 3D 理解

默认运行仓库自带示例资产（`imgs/demo_3d.glb`）；`PROMPT_3D` 指定问题。解码参数：`GEN_LEN_3D`（128）、`BLOCK_LEN_3D`（32）、`TIMESTEPS_3D`（32）。

GLB/GLTF 理解与训练完全一致地编码资产 — 内置的 `atoken_inference/glb_encode` 模块驱动 [Blender](https://www.blender.org/) 渲染 64 个彩色 / 深度视角，反投影至体素网格后交给 AToken。一次性安装 Blender，`infer_3d_sft.py` 会自动找到它：

```bash
bash tools/setup_blender.sh   # 下载 Blender 4.5 至 third_party/
```

`PATH` 上已有的 Blender 或 `BLENDER_BIN` 指定的二进制同样可用。极简容器可能缺少若干 X/GL 系统库；安装脚本会验证无头启动并打印所需的 `apt-get` 命令（无法安装系统包时，可将缺失的 `.so` 放入某目录并用 `BLENDER_LIB_DIR` 指向它）。预计算 3D token（`--token_path`）跳过编码环节，完全无需 Blender。

```bash
SECTION=3d ASSET_3D=imgs/demo_3d.glb \
  PROMPT_3D="Describe this 3D object in detail." \
  CHECKPOINT=... bash infer_1024_sft.sh
```

## 🧰 分词工具

`tools/tokenize_image.py` 将图片预编码为推理入口接受的 token 文件（编辑用 `--input_token_paths`，理解用 `--token_path`）— 适合一次编码多次复用，或直观查看模型实际"看到"的内容：

```bash
# 编码单张图片（写出 tokens/<stem>.pkl：{indices [N,8], coords [N,5]}）
python tools/tokenize_image.py --image imgs/edit_source.png --output_dir tokens/

# 以 512 长边编码整个文件夹，并输出原图|重建对照图用于目视校验
python tools/tokenize_image.py --image path/to/folder --max_edge 512 \
    --reconstruct --output_dir tokens/
```

3D 资产对应的离线编码器为 `atoken_inference/glb_encode/encode_glb_tokens.py`（`--glb ... --output ...`；需 Blender，见上文 3D 说明）。

## 🗺️ 规划

- [x] T2I / 编辑 / 图像、视频与 3D 理解的推理代码
- [ ] 模型权重
- [ ] 训练代码
- [x] 昇腾 NPU 训练与推理适配
- [ ] 技术报告

## 🙏 致谢

本项目构建于以下优秀工作之上：

- [LLaDA-2.0](https://github.com/inclusionAI/LLaDA2.0-Uni) — MoE 扩散语言模型骨干。
- [AToken (Apple ml-atoken)](https://github.com/apple/ml-atoken) — 全视觉模态共用的多码本视觉分词器；`atoken_inference/` 派生自该项目。
- [Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO) 与 [LLaDA2.0-Uni](https://github.com/inclusionAI/LLaDA2.0-Uni) — 离散扩散统一建模的开创性工作。
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni) — vendor 自上游 commit `600fe6d`，附本模型所需的本地兼容补丁。

## 📜 开源许可

本项目基于 [Apache License 2.0](LICENSE) 发布。内置的 `VeOmni/` 目录保留其上游 Apache-2.0 许可；`atoken_inference/` 派生自 Apple 的 [ml-atoken](https://github.com/apple/ml-atoken)，按 Apple 原始许可条款再分发（[atoken_inference/LICENSE](atoken_inference/LICENSE)）。

## 📖 引用

```bibtex
@misc{internluminau2,
  title  = {Intern Lumina U2},
  author = {{Intern Lumina U2 Team, Shanghai AI Laboratory}},
  year   = {2026},
  note   = {Tech report coming soon}
}
```
