# Vendored VeOmni

`VeOmni/` is vendored from [ByteDance-Seed/VeOmni](https://github.com/ByteDance-Seed/VeOmni) at commit `600fe6d`. This copy includes compatibility changes required by the Intern Lumina U2 training and checkpoint workflows.

The release-specific changes cover:

- FSDP2 parameter handling and gradient clipping;
- checkpoint saving, visibility checks, and DCP-to-HF conversion;
- data collation and worker configuration for packed multimodal samples;
- optimizer parameter grouping and training argument parsing;
- the Muon optimizer used by the optional Muon training entrypoint.

Inference uses the vendored tree through the normal package installation. The upstream VeOmni license is retained in [`VeOmni/LICENSE`](VeOmni/LICENSE).
