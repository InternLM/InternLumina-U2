"""Runtime validation for the tokenizer IDs hard-coded by inference."""

EXPECTED_SPECIAL_TOKEN_IDS = {
    "<|mask|>": 156895,
    "<system>": 157153,
    "</system>": 157154,
    "<user>": 157155,
    "<uncondition>": 157156,
    "</user>": 157157,
    "<3D_CONTEXT>": 157158,
    "<IMG_CONTEXT>": 157160,
    "<VIDEO_CONTEXT>": 157161,
    "<answer>": 157165,
    "</answer>": 157166,
    "<BOI>": 157167,
    "<EOI>": 157168,
    "<BO3d>": 157169,
    "<EO3d>": 157170,
    "<BOV>": 157171,
    "<EOV>": 157172,
}


def validate_tokenizer_contract(tokenizer) -> None:
    """Fail before generation when a checkpoint uses incompatible token IDs."""
    mismatches = []
    for token, expected_id in EXPECTED_SPECIAL_TOKEN_IDS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            mismatches.append(f"{token}: expected {expected_id}, got {actual_id}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Checkpoint tokenizer is incompatible with this inference code: " + details
        )
