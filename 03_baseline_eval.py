"""
03_baseline_eval.py
====================
Step 3 — Evaluate the BASE Mistral model on the test set BEFORE fine-tuning.

This gives the "before" baseline that Step 6 will compare against after
fine-tuning. Must be run before 04_finetune.py.

✓ CORRECT PATHS FOR YOUR PROJECT STRUCTURE:
Reads  : jsonl_format/test.jsonl
Writes : outputs/pre_finetuning_predictions.json
         outputs/pre_finetuning_metrics.json

Run:
    python 03_baseline_eval.py
    python 03_baseline_eval.py --test_path jsonl_format/test.jsonl --out_dir outputs
"""

import json
import time
import re
import argparse
from pathlib import Path
from collections import Counter

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_path",  default="jsonl_format/test.jsonl",
                    help="Path to test JSONL file (default: jsonl_format/test.jsonl)")
parser.add_argument("--out_dir",    default="outputs",
                    help="Output directory for results (default: outputs)")
parser.add_argument("--model_name", default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                    help="HuggingFace model ID for the base model")
parser.add_argument("--max_new_tokens", type=int, default=300,
                    help="Max tokens to generate per prediction")
parser.add_argument("--batch_size",     type=int, default=1,
                    help="Inference batch size (keep at 1 for 4-bit on T4)")
args = parser.parse_args()

# Create output directory
output_dir = Path(args.out_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Output directory: {output_dir.resolve()}\n")

# ─── 1. Load test set ────────────────────────────────────────────────────────
print(f"📖 Loading test set from: {args.test_path}")

test_path = Path(args.test_path)
if not test_path.exists():
    print(f"❌ Error: File not found: {args.test_path}")
    print(f"\nExpected path structure:")
    print(f"  NLP_FINAL_PROJECT/")
    print(f"  ├── jsonl_format/")
    print(f"  │   └── test.jsonl  ← Should be here")
    print(f"  ├── outputs/")
    print(f"  └── 03_baseline_eval.py")
    exit(1)

try:
    with open(args.test_path) as f:
        test_items = [json.loads(line) for line in f]
    print(f"✓ Loaded {len(test_items)} test examples\n")
except Exception as e:
    print(f"❌ Error reading file: {e}")
    exit(1)

# ─── 2. Load base model ──────────────────────────────────────────────────────
print(f"🤖 Loading base model: {args.model_name}")
print("  (this may take 2–4 minutes on first run while downloading weights)\n")

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = args.model_name,
    max_seq_length = 2048,
    dtype          = None,   # auto-detect: float16 on T4, bfloat16 on A100
    load_in_4bit   = True,
)

# Inference mode — no gradient computation needed
FastLanguageModel.for_inference(model)
print("✓ Model loaded and ready for inference.\n")

# ─── 3. Score parser ─────────────────────────────────────────────────────────
def parse_model_output(raw_text: str):
    """
    Extract score and rationale from the model's raw output string.

    The model is instructed to return:
        {"score": <int>, "rationale": "<string>"}

    We try three strategies in order:
      1. Direct json.loads() on the full output
      2. Regex to find the first {...} block and parse that
      3. Regex to find a bare "score": <digit> pattern

    Returns (score: int | None, rationale: str, parse_ok: bool)
    """
    text = raw_text.strip()

    # Strategy 1 — clean JSON output
    try:
        obj = json.loads(text)
        score     = int(obj["score"])
        rationale = obj.get("rationale", "")
        if 0 <= score <= 4:
            return score, rationale, True
    except Exception:
        pass

    # Strategy 2 — extract first {...} block
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            score     = int(obj["score"])
            rationale = obj.get("rationale", "")
            if 0 <= score <= 4:
                return score, rationale, True
        except Exception:
            pass

    # Strategy 3 — bare score digit
    match = re.search(r'"score"\s*:\s*([0-4])', text)
    if match:
        return int(match.group(1)), "", True

    # All strategies failed
    return None, text, False

# ─── 4. Run inference ────────────────────────────────────────────────────────
print(f"⚙️  Running inference on {len(test_items)} examples …")
print("  (each example takes ~5–15 seconds on T4 GPU)\n")

results   = []
t_start   = time.time()

for idx, item in enumerate(test_items):
    prompt = item["prompt_only"]

    # Tokenise
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 2048,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = args.max_new_tokens,
            do_sample      = False,        # greedy — deterministic & faster
            temperature    = 1.0,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens = True,
    ).strip()

    predicted_score, rationale, parse_ok = parse_model_output(generated)

    result = {
        "idx":             idx,
        "task":            item["task"],
        "ground_truth":    item["score"],
        "predicted_score": predicted_score,
        "predicted_rationale": rationale,
        "raw_output":      generated,
        "parse_ok":        parse_ok,
    }
    results.append(result)

    # Progress log
    gt   = item["score"]
    pred = predicted_score if predicted_score is not None else "?"
    ok   = "✓" if parse_ok else "✗ parse fail"
    match = "=" if predicted_score == gt else "≠"
    elapsed = time.time() - t_start
    print(f"  [{idx+1:2d}/{len(test_items)}] "
          f"GT={gt}  Pred={pred}  {match}  {ok}  "
          f"({elapsed:.0f}s elapsed)")

total_time = time.time() - t_start

# ─── 5. Compute metrics ──────────────────────────────────────────────────────
def compute_metrics(results, phase="pre_finetuning"):
    """
    Compute accuracy, MAE, and Quadratic Weighted Kappa (QWK).
    Entries where parsing failed are excluded from numeric metrics
    but counted separately as parse_failures.
    """
    valid = [r for r in results if r["predicted_score"] is not None]
    n_valid = len(valid)
    n_total = len(results)
    parse_failures = n_total - n_valid

    if n_valid == 0:
        print("  ⚠  No parseable predictions — check model output format.")
        return {"phase": phase, "n": n_total, "parse_failures": n_total,
                "accuracy": 0, "mae": None, "qwk": None}

    gts   = [r["ground_truth"]    for r in valid]
    preds = [r["predicted_score"] for r in valid]

    # Accuracy
    accuracy = sum(g == p for g, p in zip(gts, preds)) / n_valid

    # Mean Absolute Error
    mae = sum(abs(g - p) for g, p in zip(gts, preds)) / n_valid

    # Quadratic Weighted Kappa
    K    = 5   # scores 0–4
    conf = [[0]*K for _ in range(K)]
    for g, p in zip(gts, preds):
        conf[g][p] += 1

    W   = [[(i-j)**2 / (K-1)**2 for j in range(K)] for i in range(K)]
    gt_hist   = [sum(conf[i][j] for j in range(K)) for i in range(K)]
    pred_hist = [sum(conf[i][j] for i in range(K)) for j in range(K)]
    exp = [[gt_hist[i]*pred_hist[j]/n_valid for j in range(K)] for i in range(K)]

    num = sum(W[i][j]*conf[i][j] for i in range(K) for j in range(K))
    den = sum(W[i][j]*exp[i][j]  for i in range(K) for j in range(K))
    qwk = 1 - (num/den) if den else 0.0

    # Score distribution comparison
    pred_dist = Counter(preds)
    gt_dist   = Counter(gts)

    return {
        "phase":          phase,
        "model":          args.model_name,
        "adapter":        "none (base model)",
        "n_total":        n_total,
        "n_valid":        n_valid,
        "parse_failures": parse_failures,
        "accuracy":       round(accuracy, 4),
        "mae":            round(mae, 4),
        "qwk":            round(qwk, 4),
        "pred_distribution": dict(sorted(pred_dist.items())),
        "gt_distribution":   dict(sorted(gt_dist.items())),
        "inference_seconds": round(total_time, 1),
    }

metrics = compute_metrics(results, phase="pre_finetuning")

# ─── 6. Print report ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("PRE FINE-TUNING BASELINE RESULTS")
print("="*55)
print(f"  Model          : {args.model_name}")
print(f"  Test examples  : {metrics['n_total']}")
print(f"  Parse failures : {metrics['parse_failures']}")
print(f"  Accuracy       : {metrics['accuracy']:.4f}  "
      f"({int(metrics['accuracy']*metrics['n_valid'])}/{metrics['n_valid']} correct)")
print(f"  MAE            : {metrics['mae']:.4f}  (lower is better)")
print(f"  QWK            : {metrics['qwk']:.4f}  (higher is better, max=1.0)")

print(f"\n  Score distribution:")
print(f"  {'Score':<8} {'Ground Truth':>14} {'Predicted':>12}")
print(f"  {'-'*36}")
for s in range(5):
    gt_c   = metrics['gt_distribution'].get(s, 0)
    pred_c = metrics['pred_distribution'].get(s, 0)
    print(f"  {s:<8} {gt_c:>14} {pred_c:>12}")

print(f"\n  Total inference time: {metrics['inference_seconds']}s "
      f"({metrics['inference_seconds']/metrics['n_total']:.1f}s per example)")

# ─── 7. Save outputs ─────────────────────────────────────────────────────────
pred_path    = output_dir / "pre_finetuning_predictions.json"
metrics_path = output_dir / "pre_finetuning_metrics.json"

try:
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved predictions → {pred_path.resolve()}")
except Exception as e:
    print(f"❌ Error saving predictions: {e}")
    exit(1)

try:
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics     → {metrics_path.resolve()}")
except Exception as e:
    print(f"❌ Error saving metrics: {e}")
    exit(1)

print("\n📌 Next step: run 04_finetune.py")