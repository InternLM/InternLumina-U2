# Packed-expert model config (training)

Identical to `../internluminau2/` except for one field:

    "model_type": "llada2_moe_veomni"

That string is what selects the **packed** MoE expert layout
(`LLaDA2MoeSparseMoeBlock._setup_fuse_moe_experts`), where all experts of a layer
live in single stacked parameters:

    model.layers.N.mlp.experts.{gate_proj,up_proj,down_proj}   # [num_experts, ...]

The inference config (`model_type: "llada2_moe"`) instead builds a per-expert
`nn.ModuleList`:

    model.layers.N.mlp.experts.<i>.{gate_proj,up_proj,down_proj}.weight

The two layouts pair with the two checkpoint exports:

| use        | this config                  | checkpoint      | expert layout |
|------------|------------------------------|-----------------|---------------|
| training   | `internluminau2_packed/`     | `hf_ckpt`       | packed        |
| inference  | `internluminau2/`            | `hf_ckpt_split` | per-expert    |

Mixing them does **not** raise on its own: the loader reports
`Unexpected key in state dict: ...mlp.experts.gate_proj` for every MoE layer and
then trains from randomly initialised experts. `tasks/train_sft.py` therefore
asserts the packed model_type at startup.
