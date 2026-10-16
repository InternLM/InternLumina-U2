<div align="center">

# Intern Lumina U2：面向全视觉模态理解与图像生成的多码本扩散大语言模型

*上海人工智能实验室*

[[📑 技术报告](https://arxiv.org/abs/2610.20818)] &emsp; [[🌐 项目主页](https://internlm.github.io/InternLumina-U2/)] &emsp; [[🤗 模型](https://huggingface.co/internlm/InternLumina-U2)]

[English](./README.md) | [简体中文](./README_zh-CN.md)

</div>

![teaser](assets/teaser.png)

## 📰 新闻

- **[2026-10]** 🚀 模型权重已发布于 [Hugging Face](https://huggingface.co/internlm/InternLumina-U2)；本仓库同时补充了 **SFT 训练代码** 与可选的 **推理加速**。
- **[2026-08]** 🚀 发布推理代码（本仓库）。

## 🌟 简介

Intern Lumina U2 是一个统一多模态模型，将语言、图像、视频与 3D 纳入同一框架，以一个模型覆盖**文本问答、文生图、图像理解、图像编辑、视频理解与 3D 理解**。模型为 **16.9B 总参数、约 1.4B 激活参数的 MoE**，以高效稀疏骨干搭配基于 [AToken](https://github.com/apple/ml-atoken) 的 **8 码本全离散视觉表示**。

MMaDA、Lumina-DiMOO 等离散统一模型表明：语言与视觉一旦共享离散 token 接口，理解与生成便不再需要各自独立的模型体系。但从"能够统一"到"高性能统一"，瓶颈在于视觉内容如何被表示和预测：单码本限制了单个视觉 token 的信息容量，而离散—连续混合路线又把理解与生成拆到不同的表示与解码路径上。

Intern Lumina U2 探索了一条新路径：**多码本全离散建模**。

![overview](assets/img1.jpg)

## ✨ 亮点

**1. 一个模型覆盖全视觉模态理解、图像生成与编辑。** 文本问答、文生图、图像理解（VQA / OCR / 图表 / 数学推理）、指令式图像编辑、视频理解与 3D 理解全部运行在同一骨干与同一离散 token 接口之上。

![showcases](assets/img2.jpg)

**2. 多码本视觉表示。** [AToken](https://github.com/apple/ml-atoken) 分词器用 **8 个互补码本**共同描述每个视觉位置，在不增加序列长度的前提下扩展编码空间，保留局部纹理、文字、几何结构与高频细节。**输入侧**，每个码本保留独立 embedding，8 路逐位置 embedding 拼接后投影为单个骨干 token；**输出侧**采用*空间并行、码本深度自回归*：dLLM 骨干并行去噪多个被掩码的空间位置，多码本 AR head 按码本顺序预测每个位置的 8 个离散码。

![input](assets/img3.png)
<p align="center"><em>(a) 输入侧设计：各码本 embedding 拼接并投影为单个骨干 token。</em></p>

![output](assets/img4.png)
<p align="center"><em>(b) 输出侧设计：空间并行去噪 + 码本深度自回归 head。</em></p>

**3. 掩码扩散 MoE 骨干。** 语言骨干为 **LLaDA2.0-mini** —— 由一个预训练的自回归混合专家（MoE）语言模型转换而来的掩码扩散语言模型。**总参数 16.9B，每 token 激活约 1.4B**：20 层 Transformer 中第一层为稠密前馈网络，其余 19 层为 MoE 层，每层 256 个路由专家加 1 个共享专家，每 token 激活其中 8 个。

**4. 加速的图像生成。** **Plateau-Halton sampling with V-verify（PHV）** 将置信度平台期引导的 unmasking 与漂移触发的缓存刷新结合，**将图像生成加速 4.45 倍**。该路径训练无关且可选 —— 见[推理加速](#-推理加速)。

**5. 双硬件生态的全栈支持。** 训练、推理与评测均可运行在 **昇腾与 NVIDIA** 两套平台上，已发布的 checkpoint 覆盖两者。本仓库提供 NVIDIA 路径；昇腾侧已发布的部分见[规划](#roadmap)。

## 🏆 评测结果

已发布checkpoint 的分数。**粗体**表示在技术报告所对比的模型中为最佳；完整对比表与
基线见报告。

| 类别 | 基准 | 分数 |
|---|---|---|
| 文档与图表 | ChartQA | 86.2 |
| | CharXiv (DQ) | **85.9** |
| 感知与幻觉 | HallusionBench | **64.6** |
| | MME-C | **667** |
| | RealWorldQA | 62.9 |
| 知识与数学推理 | MMMU | 52.2 |
| | MMMU-Pro | **38.5** |
| | MathVista | 68.0 |
| | MathVision | **36.5** |
| | DynaMath | **63.6** |
| 视频理解 | Video-MME | 52.6 |
| | MLVU | 62.1 |
| | MVBench | **61.8** |
| 3D 理解 | 3D MM-Vet（总分） | **50.5** |
| 文生图 | GenEval（0–100） | **89** |
| | DPG-Bench | 87.79 |
| | TIIF-Bench（long） | **87.1** |
| | WISE | 0.56 |
| 图像编辑 | ImgEdit（1–5） | 3.82 |
| | GEdit-Bench EN / CN（0–10） | 6.06 / 5.95 |

在报告对比的统一模型中，Intern Lumina U2 在细粒度文档与图表理解（CharXiv-DQ）、幻觉鲁棒性
（HallusionBench）、知识与数学推理（MMMU-Pro、MathVision、DynaMath）、视频（MVBench）、
3D（3D MM-Vet）以及文生图的指令跟随（TIIF）上领先；在 MME-C 与 GenEval 上并列第一。

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

# 4) VeOmni —— 安装本仓库 vendor 的那一份，不是上游 VeOmni。
#    它是上游 @600fe6d 加上本项目所需的兼容补丁（见 PATCHES.md）。
#    --no-deps：依赖以 requirements.txt 为准。
pip install -e VeOmni/ --no-deps
```

说明：

- 请在 GPU 机器上运行：导入模型代码时会初始化注意力 / MoE 内核。
- 如果环境里已经装了别的 `veomni`，训练入口会拒绝启动，而不是在它上面训练。启动脚本会设置
  `PYTHONPATH=<repo>:<repo>/VeOmni`；直接调 `tasks/train_sft.py` 时请自己照做。
- `flash-attn` 只有 AToken 分词器需要（即一切涉及图像／视频／3D 编解码的任务）。
  **不需要** `liger-kernel` 与 `xformers`。
- 对象存储包（`petrel-oss-sdk` / `boto3`）可选，本地 token 文件无需它们。
- 驱动脚本会先切到仓库根目录再启动 Python，相对路径据此解析。在别处调用时设
  `LUMINA_ROOT=/path/to/InternLumina-U2`；直接调 Python 入口请用绝对路径。

## 🖥️ 昇腾 NPU

昇腾 NPU 的适配位于 [`ascend-npu-support`](https://github.com/InternLM/InternLumina-U2/tree/ascend-npu-support) 分支，而非 `main`。该分支的 LLM 骨干运行在 `torch_npu` 融合注意力内核上，融合 MoE 经由 NPU grouped matmul，因此需要 CANN 与配套的 `torch_npu`，取代上面的 CUDA 工具链与 `flash-attn`。

## 📥 模型权重

权重托管在 Hugging Face：[internlm/InternLumina-U2](https://huggingface.co/internlm/InternLumina-U2)。
同一架构在两套硬件栈上训练出的 checkpoint 各占一个子目录，按你打算怎么跑推理来下载。

```bash
pip install -U "huggingface_hub[cli]"

# 昇腾 NPU 训练的 checkpoint
huggingface-cli download internlm/InternLumina-U2 --include "ascend/*" --local-dir ckpt/ascend

# NVIDIA GPU 训练的 checkpoint
huggingface-cli download internlm/InternLumina-U2 --include "nvidia/*" --local-dir ckpt/nvidia
```

每个子目录都是开箱即用的 **split** 格式 HF checkpoint（`config.json` + safetensors +
`configuration_internluminau2.py` / `modeling_internluminau2.py` / `tokenbridge.py` /
`parallel_plan.py`），可直接作为下面的 `CHECKPOINT`。

## 🎨 推理

所有推理经由同一个驱动脚本（`infer_1024_sft.sh`），每个任务一个 section。它需要一个 **split** 格式的 HF checkpoint 目录（`.../hf_ckpt_split`，见[模型页](https://huggingface.co/internlm/InternLumina-U2)）；凡涉及图像编解码的任务，还需要位于 `atoken_inference/checkpoints/atoken-sod.pt` 的 AToken 分词器权重（或经 `ATOKEN_MODEL_PATH=...` 指定）。

底层 Python 入口（`infer_t2i_sft.py`、`infer_mmu_sft.py`、`infer_video_sft.py`、`infer_3d_sft.py`）也可直接调用，任意入口加 `--help` 查看参数。模型底层的 `generate` 是块扩散 API，请使用这些任务入口，而非标准 Transformers 的 `generate(max_length=...)` 接口。

### 需要准备的权重

两份彼此独立的产物：

1. **模型本体** —— *split* 格式的 HF checkpoint 目录，见上方[模型权重](#-模型权重)。
   如果手上是 merged 导出，用
   `python scripts/moe_convertor.py --mode split --input-path <hf_ckpt> --output-path <hf_ckpt_split>`
   转换；推理入口只接受 split 布局。
2. **AToken 视觉分词器** —— `atoken-sod.pt`，来自 Apple 的
   [ml-atoken](https://github.com/apple/ml-atoken) 发布。放到
   `atoken_inference/checkpoints/atoken-sod.pt`，或用 `ATOKEN_MODEL_PATH` 指向它。
   它**不在**本仓库里（这份权重不归我们分发，`.gitignore` 也排除了
   `atoken_inference/checkpoints/`）。凡是要编解码图像／视频／3D 的任务都需要它；纯文本生成不需要。

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

## ⚡ 推理加速

可选、训练无关、默认关闭，且**只针对图像生成** —— 编辑、文本和理解任务在本次发布中没有加速。

```bash
ACCEL=fast CHECKPOINT=/path/to/hf_ckpt_split bash infer_1024_sft.sh
```

`ACCEL` 取 `off`（默认）、`cache`、`fast`（推荐）、`quality` 之一。`off` 能逐位复现给定 seed 的
结果，其余不能，同一个 seed 不会复现之前的图。

在预设之上覆盖单个旋钮（未知的 key 会报错）：

```bash
ACCEL=fast ACCEL_SET="plateau_delta=0.17" CHECKPOINT=... bash infer_1024_sft.sh

# 或直接调用：
python infer_t2i_sft.py --accel_preset fast --accel_set plateau_delta=0.17 ...
```

`python infer_t2i_sft.py --help` 会列出每个预设暴露的旋钮。

## 🏋️ 训练

在 block-diffusion 预训练 checkpoint 之上做多任务 SFT。一个入口、一份配置、两种优化器。

```bash
# AdamW
MODEL_INIT_PATH=/path/to/pretrain/hf_ckpt \
SFT_TRAIN_PATH=data/example_sft.yaml \
EXP_NAME=./output/my_sft \
bash train_sft.sh

# Muon（矩阵参数走 Muon，embedding/head/norm/bias 走 AdamW，同一个优化器内）
MODEL_INIT_PATH=/path/to/pretrain/hf_ckpt bash train_sft_muon.sh
```

多机就是普通的 torchrun 拓扑 —— 设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
若多机任务的 `MASTER_ADDR` 还是回环地址，启动脚本会直接报错退出：否则这个故障会表现为每个节点
各自当单机训练，很难察觉。

### 初始化与续训

`MODEL_INIT_PATH` 与 `RESUME_CKPT` 互斥，`RESUME_CKPT` 优先。

- `MODEL_INIT_PATH=<pretrain hf_ckpt>` —— 开始 SFT：只搬 HF 权重，optimizer / scheduler / RNG /
  数据流全新。这是常规路径。
- `RESUME_CKPT=<dcp step dir>` —— 从完整 DCP 状态续训一个 SFT run。

有三种续训语义，因为它们回答的是不同问题：`--train.use_same_datasets`（沿用还是重建数据流）、
`--train.resume_use_new_lr`（用新 lr 还是 checkpoint 里的）、`--train.resume_reset_lr_schedule`
（调度是否从 step 0 重启）。如果重置了调度，warmup 会重跑一遍；加
`--train.resume_override_global_step <step>` 可以快进过去。Muon 侧，
`--train.muon_checkpoint_mode resume` 才会恢复优化器状态 —— 默认的 `model_only` 是有意从新的
Muon 开始。

### 数据

一个列出 `.jsonl` 标注文件的 manifest YAML；视觉内容以预先编码好的 AToken code 引用，不是像素。
参见 [`data/example_sft.yaml`](data/example_sft.yaml) 与
[`data/examples/example_sft.jsonl`](data/examples/example_sft.jsonl)。

想要一份能直接跑起来的示例数据（随机视觉 token，只验证链路、不学东西）：

```bash
python tools/make_example_data.py --output_dir data/example_run
SFT_TRAIN_PATH=data/example_run/train.yaml \
MODEL_INIT_PATH=/path/to/pretrain/hf_ckpt bash train_sft.sh
```

### Checkpoint

训练写出 DCP checkpoint 以及可选的 HF 权重。从 DCP step 目录转成推理驱动能接受的格式：

```bash
python VeOmni/scripts/mereg_dcp_to_hf.py --load-dir <dcp-step-dir> --save-dir <hf_ckpt>
python scripts/moe_convertor.py --mode split \
    --input-path <hf_ckpt> --output-path <hf_ckpt_split>
```

两步都不能省，且顺序固定。第一步合并出来的目录只是**中间产物**，必须再跑 split，并让推理指向
split 的输出（`<hf_ckpt_split>`）。推理入口只接受 split 布局，遇到合并的（packed-MoE）
checkpoint 会直接拒绝加载，而不会错误地加载。

split 这一步不能多个任务并发写同一个输出目录。

### 说明

- `configs/sft_packing.yaml` 是默认配置；传给启动脚本的任何尾随 CLI 参数都会覆盖它。
- 训练必须 import 本仓库 **vendor 的** `VeOmni/`，而不是环境里其它的 `veomni`。启动脚本已经把仓库根和
  `VeOmni/` 都加进 `PYTHONPATH`；如果你直接调 `tasks/train_sft.py`，请自己照做。

  ```bash
  export PYTHONPATH=<repo>:<repo>/VeOmni
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

<a id="roadmap"></a>

## 🗺️ 规划

- [x] T2I / 编辑 / 图像、视频与 3D 理解的推理代码
- [x] 模型权重
- [x] 多任务 SFT 训练代码
- [x] 训练无关的推理加速
- [x] 技术报告
- [x] 昇腾 NPU 推理 —— 见 [`ascend-npu-support`](https://github.com/InternLM/InternLumina-U2/tree/ascend-npu-support) 分支
- [ ] 昇腾 NPU 训练栈
- [ ] 用于在线体验的 Hugging Face Space

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
@article{internluminau2,
  title   = {Intern Lumina U2: A Multi-Codebook Diffusion Large Language Model for
             Omni-Visual Understanding and Image Generation},
  author  = {{Intern Lumina U2 Team, Shanghai AI Laboratory}},
  journal = {arXiv preprint arXiv:2610.20818},
  year    = {2026}
}
```
