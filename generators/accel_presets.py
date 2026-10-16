# -*- coding: utf-8 -*-
"""Acceleration presets for image generation.

Training-free, opt-in, and image generation only. ``off`` enables nothing and
reproduces the released output; the other presets trade some reproducibility
(and, for ``fast`` / ``quality``, some decode steps) for speed, so a given seed
will not reproduce an earlier image.

Select one with ``ACCEL=<name>`` on the driver or ``--accel_preset <name>`` on
``infer_t2i_sft.py``, and override individual knobs with ``--accel_set k=v``.
``plateau_delta > 0`` always implies ``halton=True``; they are a single unit and
cannot be set apart.
"""

ACCEL_PRESETS = {
    "off": dict(
        ar_kv_cache=False,
        cfg_batch=False,
        ml_cache=False,
    ),
    "cache": dict(
        ar_kv_cache=True,
        cfg_batch=True,
        ml_cache=True,
        cache_ratio=0.9,
        refresh_interval=12,
        vverify_delta=0.5,
        warmup_ratio=0.3,
    ),
    # Recommended.
    "fast": dict(
        ar_kv_cache=True,
        cfg_batch=True,
        ml_cache=True,
        cache_ratio=0.9,
        refresh_interval=12,
        vverify_delta=0.5,
        warmup_ratio=0.3,
        plateau_delta=0.18,
        plateau_floor=0.005,
        plateau_patience=5,
        plateau_min_samples=4,
        conf_ema_beta=0.7,
    ),
    # More conservative early commit.
    "quality": dict(
        ar_kv_cache=True,
        cfg_batch=True,
        ml_cache=True,
        cache_ratio=0.9,
        refresh_interval=12,
        vverify_delta=0.5,
        warmup_ratio=0.3,
        plateau_delta=0.16,
        plateau_floor=0.015,
        plateau_patience=5,
        plateau_min_samples=4,
        conf_ema_beta=0.7,
    ),
}

ACCEL_PRESET_NAMES = tuple(ACCEL_PRESETS)


def resolve_accel_preset(name):
    """Return a fresh kwargs dict for ``name``.

    ``plateau_delta > 0`` implies ``halton=True``; the caller must not be able to
    separate them.
    """
    if name not in ACCEL_PRESETS:
        raise ValueError(
            f"unknown accel preset {name!r}; expected one of {list(ACCEL_PRESET_NAMES)}"
        )
    cfg = dict(ACCEL_PRESETS[name])
    cfg["halton"] = cfg.get("plateau_delta", 0.0) > 0
    return cfg
