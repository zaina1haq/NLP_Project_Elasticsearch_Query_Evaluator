"""
02_format_prompts.py
=====================
Step 2 — Format the labeled dataset into Mistral instruction-tuning prompts.

Reads  : es_eval_train.json / es_eval_val.json / es_eval_test.json
Writes : formatted/train.jsonl / formatted/val.jsonl / formatted/test.jsonl

Each output line is a JSON object with one key:
    "text" : the full <s>[INST]…[/INST] … </s> string

The test split keeps both the formatted "text" AND the raw fields so that
evaluation scripts can compare predicted vs ground-truth score without
having to reload the original JSON.

Run:
    python 02_format_prompts.py
    python 02_format_prompts.py --data_dir /path/to/splits --out_dir /path/to/formatted
"""

import json
import argparse
import os
from pathlib import Path

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="json_format",
                    help="Folder containing es_eval_train/val/test.json")

parser.add_argument("--out_dir",  default="jsonl_format",
                    help="Where to write the .jsonl files")

parser.add_argument("--max_rubric_chars", type=int, default=900,
                    help="Truncate each rubric level to this many chars "
                         "(keeps prompts from exceeding the 2048-token window)")

args = parser.parse_args()

Path(args.out_dir).mkdir(parents=True, exist_ok=True)

# ─── Mistral special tokens ───────────────────────────────────────────────────
BOS  = "<s>"
EOS  = "</s>"
B_INST, E_INST = "[INST]", "[/INST]"

# ─── System preamble (injected once inside the first [INST] block) ────────────
# Mistral-Instruct v0.2 has no dedicated system-token; the convention is to
# embed the system message as the very first line of the [INST] block.
SYSTEM = (
    "You are an expert Elasticsearch query evaluator. "
    "Given a task description, a reference (correct) query, a student submission, "
    "and a scoring rubric, you must:\n"
    "1. Evaluate the submission against the reference using the rubric.\n"
    "2. Output ONLY a JSON object with exactly two keys:\n"
    '   "score"    : an integer from 0 to 4\n'
    '   "rationale": a single string explaining the score, citing each of the '
    "four rubric dimensions [Query Structure], [Field Validity], "
    "[Operator Correctness], and [Task Alignment].\n"
    "Do not output anything outside the JSON object."
)

# ─── Rubric formatter ─────────────────────────────────────────────────────────
def format_rubric(rubric: dict, max_chars: int) -> str:
    """
    Render the rubric as a compact numbered list.
    Each level is truncated to `max_chars` so the prompt stays within the
    model's context window even for our verbose multi-dimensional rubric.
    """
    lines = []
    for level in sorted(rubric.keys(), key=int):
        text = rubric[level]
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        lines.append(f"  {level}: {text}")
    return "\n".join(lines)

# ─── Single-entry formatter ───────────────────────────────────────────────────
def build_prompt(entry: dict, max_rubric_chars: int) -> str:
    """
    Build the instruction portion of the prompt (everything inside [INST]…[/INST]).
    Returns the full string WITHOUT the expected output — used at inference time.
    """
    ref_str = json.dumps(entry["reference"],  indent=2, ensure_ascii=False)
    sub_str = json.dumps(entry["submission"], indent=2, ensure_ascii=False)
    rubric_str = format_rubric(entry["rubric"], max_rubric_chars)

    instruction = (
        f"{SYSTEM}\n\n"
        f"### Task\n{entry['task']}\n\n"
        f"### Reference query (correct answer)\n```json\n{ref_str}\n```\n\n"
        f"### Student submission (to evaluate)\n```json\n{sub_str}\n```\n\n"
        f"### Scoring rubric\n{rubric_str}\n\n"
        f"Evaluate the submission and respond with a JSON object."
    )
    return instruction

def build_expected_output(entry: dict) -> str:
    """
    Build the expected model output: a JSON object with score + rationale.
    Rationale pipes are converted to newlines for readability inside the string.
    """
    rationale = entry["rationale"].replace(" | ", "\n")
    output = {"score": entry["score"], "rationale": rationale}
    return json.dumps(output, ensure_ascii=False)

def build_full_text(entry: dict, max_rubric_chars: int) -> str:
    """
    Assemble the complete training string in Mistral chat format:
        <s>[INST] … [/INST] … </s>
    """
    instruction = build_prompt(entry, max_rubric_chars)
    response    = build_expected_output(entry)
    return f"{BOS}{B_INST} {instruction} {E_INST} {response}{EOS}"

# ─── Process splits ──────────────────────────────────────────────────────────
SPLITS = {
    "train": False,   # train/val: only the formatted text is needed
    "val":   False,
    "test":  True,    # test: keep raw fields for evaluation
}

stats = {}

for split, keep_raw in SPLITS.items():
    in_path  = os.path.join(args.data_dir, f"es_eval_{split}.json")
    out_path = os.path.join(args.out_dir,  f"{split}.jsonl")

    if not os.path.exists(in_path):
        print(f"  ⚠  {in_path} not found — skipping {split} split.")
        continue

    with open(in_path) as f:
        entries = json.load(f)

    token_lengths = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for entry in entries:
            full_text = build_full_text(entry, args.max_rubric_chars)

            # Rough token estimate: ~4 chars per token (good enough for a
            # sanity check without loading the actual tokenizer here)
            token_lengths.append(len(full_text) // 4)

            record = {"text": full_text}

            if keep_raw:
                # Attach raw fields so the eval script can extract ground truth
                record["task"]       = entry["task"]
                record["reference"]  = entry["reference"]
                record["submission"] = entry["submission"]
                record["score"]      = entry["score"]
                record["rationale"]  = entry["rationale"]
                # Also store just the inference prompt (no answer) for inference
                record["prompt_only"] = (
                    f"{BOS}{B_INST} "
                    f"{build_prompt(entry, args.max_rubric_chars)} "
                    f"{E_INST}"
                )

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_tok = sum(token_lengths) / len(token_lengths)
    max_tok = max(token_lengths)
    stats[split] = {
        "n": len(entries),
        "avg_tokens_est": round(avg_tok),
        "max_tokens_est": max_tok,
    }
    print(f"  ✓  {split:5s} → {out_path}  "
          f"({len(entries)} examples, "
          f"avg ~{round(avg_tok)} tokens, "
          f"max ~{max_tok} tokens)")

# ─── Sanity-check: print one formatted example ───────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE FORMATTED ENTRY (train split, first example)")
print("=" * 60)

sample_path = os.path.join(args.out_dir, "train.jsonl")
if os.path.exists(sample_path):
    with open(sample_path) as f:
        sample = json.loads(f.readline())
    text = sample["text"]
    # Print the instruction block and the response separately for readability
    if E_INST in text:
        inst_part, resp_part = text.split(E_INST, 1)
        print("\n── INSTRUCTION ──────────────────────────────────────────")
        print(inst_part.replace(BOS + B_INST + " ", "").strip()[:1200])
        print("\n── EXPECTED RESPONSE ────────────────────────────────────")
        print(resp_part.replace(EOS, "").strip())
    else:
        print(text[:1500])

# ─── Token length warnings ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOKEN LENGTH SUMMARY (estimated, ÷4 chars/token)")
print("=" * 60)
for split, s in stats.items():
    flag = "  ⚠  exceeds 2048!" if s["max_tokens_est"] > 2048 else ""
    print(f"  {split:5s}  n={s['n']:4d}  "
          f"avg={s['avg_tokens_est']:4d}  "
          f"max={s['max_tokens_est']:4d}{flag}")

print(f"\nFormatted files written to: {os.path.abspath(args.out_dir)}/")
print("Next step: run 03_baseline_eval.py")
