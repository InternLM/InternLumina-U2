#!/usr/bin/env python3
"""Check an SFT dataset before training on it. CPU only, no GPU or weights.

    python tests/test_sft_dataset.py
    python tests/test_sft_dataset.py --tokenizer /path/to/hf_ckpt_split

Validates what usually goes wrong first: that a record maps to the expected
turns, that it yields supervised positions, and that the checkpoint's tokenizer
carries the special-token ids the data pipeline assumes.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.dataset import xllmx_dataset_sft_multitask_blkpad as ds  # noqa: E402
from utils import media_layout  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


class StubTokenizer:
    """Maps each word to a distinct id above the special-token range."""

    all_special_ids = list(ds.SPECIAL_TOKEN_IDS.values())

    def __init__(self):
        self._vocab = {}

    def _id(self, word):
        if word in ds.SPECIAL_TOKEN_IDS:
            return ds.SPECIAL_TOKEN_IDS[word]
        return self._vocab.setdefault(word, 1000 + len(self._vocab))

    def __call__(self, text, truncation=True, max_length=None, padding=False, return_tensors=None):
        ids = [self._id(w) for w in (text or "").split()][: max_length or None]
        return type("Enc", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

    def encode(self, text, add_special_tokens=False):
        return [self._id(text)]

    def convert_ids_to_tokens(self, idx):
        for tok, i in ds.SPECIAL_TOKEN_IDS.items():
            if i == idx:
                return tok
        return f"<unk:{idx}>"


def test_flat_schema_normalization():
    """Flat records map to messages; the messages schema stays untouched."""
    n = ds._normalize_flat_item

    # t2i: unchanged behaviour (token_path is the generated image).
    out = n({"task_type": "t2i", "token_path": "a.pt", "prompt": "a cat",
             "image_width": 256})
    user, asst = out["messages"]
    check("flat t2i user carries the prompt",
          user["content"][0] == {"type": "text", "text": "a cat"}, str(user))
    check("flat t2i assistant carries the image",
          asst["content"][0]["type"] == "image_url"
          and asst["content"][0]["token"]["token_path"] == "a.pt", str(asst))
    check("flat t2i forwards image metadata",
          asst["content"][0]["token"].get("image_width") == 256)

    # under: token_path is the *input*, the answer is text.
    out = n({"task_type": "under", "token_path": "img.pt",
             "prompt": "what is this?", "answer": "a cat"})
    user, asst = out["messages"]
    check("flat under puts the media in the user turn",
          user["content"][0]["type"] == "image_url", str(user))
    check("flat under keeps the question after the media",
          user["content"][1] == {"type": "text", "text": "what is this?"})
    check("flat under answers in text", asst["content"] == "a cat")

    # Media type follows the task type.
    for task, want in (("under_video", "video_url"), ("under_3d", "3d_url")):
        out = n({"task_type": task, "token_path": "m.pt", "prompt": "q", "answer": "a"})
        check(f"flat {task} uses {want}",
              out["messages"][0]["content"][0]["type"] == want)

    # text: no media at all.
    out = n({"task_type": "text", "prompt": "hi", "answer": "hello"})
    check("flat text has no media part",
          all(p["type"] == "text" for p in out["messages"][0]["content"]))

    # single_edit needs both images, and says which is which.
    out = n({"task_type": "single_edit", "input_token_path": "src.pt",
             "token_path": "dst.pt", "prompt": "make it blue"})
    user, asst = out["messages"]
    check("flat single_edit conditions on input_token_path",
          user["content"][0]["token"]["token_path"] == "src.pt", str(user))
    check("flat single_edit outputs token_path",
          asst["content"][0]["token"]["token_path"] == "dst.pt")
    try:
        n({"task_type": "single_edit", "token_path": "dst.pt", "prompt": "x"})
        check("flat single_edit without a source is rejected", False)
    except ValueError as e:
        check("flat single_edit without a source is rejected",
              "input_token_path" in str(e), str(e))

    # A missing answer is an error, not an empty supervised turn.
    try:
        n({"task_type": "under", "token_path": "img.pt", "prompt": "q"})
        check("flat under without an answer is rejected", False)
    except ValueError as e:
        check("flat under without an answer is rejected", "answer" in str(e), str(e))

    # Records that already use messages are returned as-is.
    msg_item = {"task_type": "under", "messages": [{"role": "user", "content": "q"}]}
    check("messages records pass through untouched", n(msg_item) is msg_item)


def test_flat_under_reaches_supervision():
    """End-to-end: a flat `under` record produces supervised text positions."""
    import tempfile

    tok = StubTokenizer()
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/img.pt"
        grid = 4
        coords = torch.zeros(grid * grid, 5, dtype=torch.long)
        k = 0
        for y in range(grid):
            for x in range(grid):
                coords[k, 2], coords[k, 3] = y, x
                k += 1
        torch.save({"indices": torch.randint(0, 4096, (grid * grid, 8)),
                    "coords": coords}, path)

        proc = ds.ItemProcessorBlkPad(tokenizer=tok, max_len=4096, block_size=32,
                                      max_text_length=512)
        d = proc.process_item({
            "id": "f0", "task_type": "under", "system_prompt": "answer it",
            "token_path": path, "prompt": "what is this",
            "answer": "a synthetic pattern",
        })[0]

    tp, labels = d["type_position_ids"], d["labels"]
    sup = (tp == ds.TYPE_TEXT_ANSWER) & (labels[:, 0] != -100)
    check("flat under yields supervised text positions", int(sup.sum()) > 0,
          f"n={int(sup.sum())}")
    check("flat under carries the input media",
          int((tp == ds.TYPE_MEDIA_PROMPT).sum()) > 0,
          f"n={int((tp == ds.TYPE_MEDIA_PROMPT).sum())}")


def test_task_type_surface():
    supported = ds.TEXT_UNDERSTANDING_TASKS | ds.IMAGE_OUTPUT_TASKS
    expected = {"text", "under", "under_think", "under_video", "under_3d",
                "t2i", "t2i_think", "single_edit"}
    check("supported task types are exactly the released set", supported == expected,
          f"got {sorted(supported)}")
    check("uncondition tasks are image-output only",
          ds.UNCONDITION_TASKS == {"t2i", "t2i_think", "single_edit"},
          str(sorted(ds.UNCONDITION_TASKS)))

    proc = ds.ItemProcessorBlkPad.__new__(ds.ItemProcessorBlkPad)
    try:
        ds.ItemProcessorBlkPad.process_item(proc, {"task_type": "not_a_task_type"})
    except ValueError as e:
        check("an unsupported task type is rejected",
              "Unsupported SFT task_type" in str(e), str(e)[:80])
    except Exception as e:  # noqa: BLE001
        check("an unsupported task type is rejected", False,
              f"raised {type(e).__name__} instead of ValueError: {e}")
    else:
        check("an unsupported task type is rejected", False, "no error raised")


def test_media_layout_shapes():
    toks = torch.arange(6 * 4 * 8, dtype=torch.long).reshape(24, 8)
    out = media_layout.add_break_line_multi_codebook(toks, 6, 4, new_number=5001)
    check("row separators give H*(W+1) rows", out.shape == (6 * 5, 8), str(out.shape))
    check("separator lands at the end of each row",
          bool((out.view(6, 5, 8)[:, -1, :] == 5001).all()))


def test_with_real_tokenizer(tokenizer_path):
    """End-to-end `process_item` for every task type, with a real tokenizer.

    The stub tokenizer above is built *from* SPECIAL_TOKEN_IDS, so it cannot
    detect a wrong id -- it is circular by construction. This check loads the
    checkpoint's own tokenizer and asserts the contract against it, then builds
    one sample of every supported task type from a locally written token file.
    """
    import os
    import tempfile

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"  tokenizer: {tokenizer_path} (vocab={len(tok)})")

    try:
        ds.verify_sft_special_token_ids(tok)
        check("real tokenizer satisfies the training token contract", True)
    except Exception as e:  # noqa: BLE001
        check("real tokenizer satisfies the training token contract", False, str(e)[:200])

    from utils.tokenizer_contract import validate_tokenizer_contract
    try:
        validate_tokenizer_contract(tok)
        check("real tokenizer satisfies the inference token contract", True)
    except Exception as e:  # noqa: BLE001
        check("real tokenizer satisfies the inference token contract", False, str(e)[:200])

    # The whole v0/v1 split rests on this: 156895 must be a *pretrained* mask
    # embedding, and the pre-release 157147 must be a reserved (dead) row.
    check("TEXT_MASK_ID decodes to <|mask|>",
          tok.convert_ids_to_tokens(ds.TEXT_MASK_ID) == "<|mask|>",
          repr(tok.convert_ids_to_tokens(ds.TEXT_MASK_ID)))
    check("the pre-release 157147 is a reserved token, not a real mask",
          "reserved" in str(tok.convert_ids_to_tokens(157147)),
          repr(tok.convert_ids_to_tokens(157147)))

    tmp = tempfile.mkdtemp()
    h = w = 8
    n = h * w
    idx = torch.randint(0, 4096, (n, 8), dtype=torch.long)
    co = torch.zeros(n, 5, dtype=torch.long)
    k = 0
    for y in range(h):
        for x in range(w):
            co[k, 2], co[k, 3] = y, x
            k += 1
    img = os.path.join(tmp, "tokens.pt")
    torch.save({"indices": idx, "coords": co}, img)   # the loader uses torch.load
    meta = {"token_path": img, "token_length": n,
            "image_width": w * 16, "image_height": h * 16}

    proc = ds.ItemProcessorBlkPad(
        tokenizer=tok, max_len=8192, block_size=32, max_text_length=512
    )
    img_part = {"type": "image_url", "token": dict(meta)}
    cases = {
        "text": [{"role": "user", "content": [{"type": "text", "text": "Why is the sky blue?"}]},
                 {"role": "assistant", "content": "Rayleigh scattering."}],
        "t2i": [{"role": "user", "content": [{"type": "text", "text": "a red fox"}]},
                {"role": "assistant", "content": [dict(img_part)]}],
        "under": [{"role": "user", "content": [dict(img_part),
                   {"type": "text", "text": "<IMG_CONTEXT>\nWhat animal?"}]},
                  {"role": "assistant", "content": "A fox."}],
        "single_edit": [{"role": "user", "content": [dict(img_part),
                         {"type": "text", "text": "<IMG_CONTEXT>\nmake it night"}]},
                        {"role": "assistant", "content": [dict(img_part)]}],
        "under_think": [{"role": "user", "content": [dict(img_part),
                         {"type": "text", "text": "<IMG_CONTEXT>\nWhat animal?"}]},
                        {"role": "assistant", "content": "<think>Bushy tail.</think>A fox."}],
        "t2i_think": [{"role": "user", "content": [{"type": "text", "text": "a red fox"}]},
                      {"role": "assistant", "content": "<think>Snow, dusk.</think>"},
                      {"role": "assistant", "content": [dict(img_part)]}],
    }
    for task, messages in cases.items():
        item = {"task_type": task, "system_prompt": "S.", "messages": messages}
        try:
            out = proc.process_item(dict(item))[0]
            ii, lab = out["input_ids"], out["labels"]
            sane = (
                ii.shape[0] > 0
                and ii.shape == lab.shape
                and int((lab[:, 0] != -100).sum()) > 0
                and out["type_position_ids"].shape[0] == ii.shape[0]
            )
            check(f"real-tokenizer {task}: builds a supervised sample", sane,
                  f"S={ii.shape[0]} supervised={int((lab[:, 0] != -100).sum())}")
            proc.predict_item_token_length(dict(item))
        except Exception as e:  # noqa: BLE001
            check(f"real-tokenizer {task}: builds a supervised sample", False,
                  f"{type(e).__name__}: {str(e)[:140]}")


def test_length_estimate_never_under(tokenizer_path):
    """`predict_item_token_length` must never come in under the real length.

    The packer trusts this estimate before the sample is loaded, and a low
    estimate shows up much later as an oversize pack that gets silently skipped.
    Small images hide the problem behind LENGTH_ESTIMATE_MARGIN, so check the
    real output sizes -- 1024px single_edit carries two 4096-token images and is
    the tightest case.
    """
    import os
    import tempfile

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tmp = tempfile.mkdtemp()

    def token_file(px):
        g = px // 16
        n = g * g
        co = torch.zeros(n, 5, dtype=torch.long)
        k = 0
        for y in range(g):
            for x in range(g):
                co[k, 2], co[k, 3] = y, x
                k += 1
        p = os.path.join(tmp, f"i{px}.pt")
        torch.save({"indices": torch.randint(0, 4096, (n, 8), dtype=torch.long),
                    "coords": co}, p)
        return {"token_path": p, "token_length": n, "image_width": px, "image_height": px}

    random.seed(0)
    proc = ds.ItemProcessorBlkPad(tokenizer=tok, max_len=65536, block_size=32,
                                  max_text_length=2048)
    for px in (256, 512, 1024):
        meta = token_file(px)
        img = {"type": "image_url", "token": dict(meta)}
        for task, msgs in (
            ("t2i", [{"role": "user", "content": [{"type": "text", "text": "a red fox " * 20}]},
                     {"role": "assistant", "content": [dict(img)]}]),
            ("single_edit", [{"role": "user", "content": [dict(img),
                              {"type": "text", "text": "<IMG_CONTEXT>\nnight " * 10}]},
                             {"role": "assistant", "content": [dict(img)]}]),
            ("under", [{"role": "user", "content": [dict(img),
                        {"type": "text", "text": "<IMG_CONTEXT>\nDescribe " * 10}]},
                       {"role": "assistant", "content": "A fox. " * 50}]),
        ):
            item = {"task_type": task, "system_prompt": "S.", "messages": msgs}
            real = proc.process_item(dict(item))[0]["input_ids"].shape[0]
            est = proc.predict_item_token_length(dict(item))
            check(f"{px}px {task}: length estimate >= actual ({est:.0f} >= {real})",
                  est >= real, f"UNDER by {real - est:.0f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", default=None,
                    help="checkpoint directory; enables the tokenizer checks")
    a = ap.parse_args()

    print("flat schema");            test_flat_schema_normalization()
    print("flat record end-to-end"); test_flat_under_reaches_supervision()
    print("task types");             test_task_type_surface()
    print("media layout");           test_media_layout_shapes()
    if a.tokenizer:
        print("tokenizer");          test_with_real_tokenizer(a.tokenizer)
        print("length estimate");    test_length_estimate_never_under(a.tokenizer)
    else:
        print("tokenizer: SKIPPED (pass --tokenizer <checkpoint-dir> to run)")
    print()
    if FAILURES:
        print(f"FAIL ({len(FAILURES)}): {FAILURES}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
