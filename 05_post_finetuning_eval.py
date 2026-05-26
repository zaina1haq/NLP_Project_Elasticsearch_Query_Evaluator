"""
05_post_finetuning_eval.py
===========================
Step 5 — Evaluate the FINE-TUNED model on the same test set, then
produce a full before/after comparison report.

Reads  : formatted/test.jsonl
         outputs/pre_finetuning_metrics.json
         outputs/finetuned_model/lora_adapter/   (LoRA weights)
Writes : outputs/post_finetuning_predictions.json
         outputs/post_finetuning_metrics.json
         outputs/comparison_report.json           ← before vs after delta

Run:
    python 05_post_finetuning_eval.py
    python 05_post_finetuning_eval.py \
        --adapter outputs/finetuned_model/lora_adapter \
        --test_path formatted/test.jsonl \
        --pre_metrics outputs/pre_finetuning_metrics.json
"""

import json
import time
import re
import argparse
from pathlib import Path
from collections import Counter

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--adapter",     default="outputs/finetuned_model/lora_adapter",
                    help="Path to the saved LoRA adapter folder")
parser.add_argument("--test_path",   default="jsonl_format/test.jsonl")
parser.add_argument("--pre_metrics", default="outputs/pre_finetuning_metrics.json",
                    help="Metrics JSON produced by 03_baseline_eval.py")
parser.add_argument("--out_dir",     default="outputs")
parser.add_argument("--max_new_tokens", type=int, default=300)
args = parser.parse_args()

Path(args.out_dir).mkdir(parents=True, exist_ok=True)

# ─── 1. Load test set ────────────────────────────────────────────────────────
print(f"Loading test set from: {args.test_path}")
with open(args.test_path) as f:
    test_items = [json.loads(line) for line in f]
print(f"  {len(test_items)} test examples")

# ─── 2. Load pre-finetuning baseline metrics ─────────────────────────────────
print(f"\nLoading baseline metrics from: {args.pre_metrics}")
with open(args.pre_metrics) as f:
    pre = json.load(f)
print(f"  Baseline accuracy : {pre['accuracy']:.4f}")
print(f"  Baseline MAE      : {pre['mae']:.4f}")
print(f"  Baseline QWK      : {pre['qwk']:.4f}")

# ─── 3. Load fine-tuned model ────────────────────────────────────────────────
print(f"\nLoading fine-tuned model from adapter: {args.adapter}")

from unsloth import FastLanguageModel
import torch

# Load the base model first (same config as training)
BASE_MODEL     = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,
    load_in_4bit   = True,
)

# Apply the saved LoRA adapter on top
from peft import PeftModel
model = PeftModel.from_pretrained(model, args.adapter)

# Switch to inference mode — disables dropout, enables Unsloth's fast path
FastLanguageModel.for_inference(model)
print("  Fine-tuned model loaded and ready for inference.")

# ─── 4. Score parser (same as Step 3) ────────────────────────────────────────
def parse_model_output(raw_text: str):
    """
    Extract score and rationale from the model's raw output.
    Three fallback strategies in order:
      1. Direct json.loads()
      2. Regex on first {...} block
      3. Regex on bare "score": <digit>
    """
    text = raw_text.strip()

    try:
        obj = json.loads(text)
        score     = int(obj["score"])
        rationale = obj.get("rationale", "")
        if 0 <= score <= 4:
            return score, rationale, True
    except Exception:
        pass

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

    match = re.search(r'"score"\s*:\s*([0-4])', text)
    if match:
        return int(match.group(1)), "", True

    return None, text, False

# ─── 5. Run inference ────────────────────────────────────────────────────────
print(f"\nRunning inference on {len(test_items)} examples …\n")

results = []
t_start = time.time()

for idx, item in enumerate(test_items):
    prompt = item["prompt_only"]

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = args.max_new_tokens,
            do_sample      = False,
            temperature    = 1.0,
            pad_token_id   = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens = True,
    ).strip()

    predicted_score, rationale, parse_ok = parse_model_output(generated)

    result = {
        "idx":                 idx,
        "task":                item["task"],
        "ground_truth":        item["score"],
        "predicted_score":     predicted_score,
        "predicted_rationale": rationale,
        "raw_output":          generated,
        "parse_ok":            parse_ok,
    }
    results.append(result)

    gt    = item["score"]
    pred  = predicted_score if predicted_score is not None else "?"
    match = "✓" if predicted_score == gt else "✗"
    delta = f"(Δ={predicted_score - gt:+d})" if predicted_score is not None else ""
    print(f"  [{idx+1:2d}/{len(test_items)}] GT={gt}  Pred={pred}  {match} {delta}")

total_time = time.time() - t_start

# ─── 6. Compute metrics ──────────────────────────────────────────────────────
def compute_metrics(results, phase, adapter_path):
    valid          = [r for r in results if r["predicted_score"] is not None]
    n_valid        = len(valid)
    n_total        = len(results)
    parse_failures = n_total - n_valid

    if n_valid == 0:
        return {"phase": phase, "n": n_total, "parse_failures": n_total,
                "accuracy": 0, "mae": None, "qwk": None}

    gts   = [r["ground_truth"]    for r in valid]
    preds = [r["predicted_score"] for r in valid]

    accuracy = sum(g == p for g, p in zip(gts, preds)) / n_valid
    mae      = sum(abs(g - p) for g, p in zip(gts, preds)) / n_valid

    K    = 5
    conf = [[0]*K for _ in range(K)]
    for g, p in zip(gts, preds):
        conf[g][p] += 1
    W        = [[(i-j)**2 / (K-1)**2 for j in range(K)] for i in range(K)]
    gt_hist  = [sum(conf[i][j] for j in range(K)) for i in range(K)]
    ph_hist  = [sum(conf[i][j] for i in range(K)) for j in range(K)]
    exp      = [[gt_hist[i]*ph_hist[j]/n_valid for j in range(K)] for i in range(K)]
    num      = sum(W[i][j]*conf[i][j] for i in range(K) for j in range(K))
    den      = sum(W[i][j]*exp[i][j]  for i in range(K) for j in range(K))
    qwk      = 1 - (num/den) if den else 0.0

    # Per-score breakdown: how many correct per score level
    per_score = {}
    for s in range(5):
        gt_s      = [r for r in valid if r["ground_truth"] == s]
        correct_s = [r for r in gt_s  if r["predicted_score"] == s]
        per_score[str(s)] = {
            "total":   len(gt_s),
            "correct": len(correct_s),
            "acc":     round(len(correct_s)/len(gt_s), 4) if gt_s else None,
        }

    return {
        "phase":             phase,
        "model":             BASE_MODEL,
        "adapter":           str(adapter_path),
        "n_total":           n_total,
        "n_valid":           n_valid,
        "parse_failures":    parse_failures,
        "accuracy":          round(accuracy, 4),
        "mae":               round(mae, 4),
        "qwk":               round(qwk, 4),
        "per_score_accuracy": per_score,
        "pred_distribution": dict(sorted(Counter(preds).items())),
        "gt_distribution":   dict(sorted(Counter(gts).items())),
        "inference_seconds": round(total_time, 1),
    }

post = compute_metrics(results, "post_finetuning", args.adapter)

# ─── 7. Build comparison report ───────────────────────────────────────────────
def delta_str(post_val, pre_val, higher_better=True):
    """Format a metric delta with direction arrow."""
    d = post_val - pre_val
    if higher_better:
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "—")
    else:
        arrow = "▲" if d < 0 else ("▼" if d > 0 else "—")
    return f"{d:+.4f} {arrow}"

comparison = {
    "pre_finetuning": {
        "accuracy": pre["accuracy"],
        "mae":      pre["mae"],
        "qwk":      pre["qwk"],
    },
    "post_finetuning": {
        "accuracy": post["accuracy"],
        "mae":      post["mae"],
        "qwk":      post["qwk"],
    },
    "delta": {
        "accuracy": round(post["accuracy"] - pre["accuracy"], 4),
        "mae":      round(post["mae"]      - pre["mae"],      4),
        "qwk":      round(post["qwk"]      - pre["qwk"],      4),
    },
    "score_distribution": {
        "ground_truth":   post["gt_distribution"],
        "pre_predicted":  pre["pred_distribution"],
        "post_predicted": post["pred_distribution"],
    },
    "per_score_accuracy_post": post["per_score_accuracy"],
    "notes": {
        "base_over_predicts_4": (
            pre["pred_distribution"].get("4", 0) > pre["gt_distribution"].get("4", 0)
        ),
        "base_under_predicts_1": (
            pre["pred_distribution"].get("1", 0) < pre["gt_distribution"].get("1", 0)
        ),
    }
}

# ─── 8. Print report ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BEFORE vs AFTER FINE-TUNING — COMPARISON REPORT")
print("="*60)
print(f"\n  {'Metric':<12} {'Before':>10} {'After':>10} {'Delta':>14}")
print(f"  {'-'*48}")
print(f"  {'Accuracy':<12} {pre['accuracy']:>10.4f} {post['accuracy']:>10.4f} "
      f"  {delta_str(post['accuracy'], pre['accuracy'])}")
print(f"  {'MAE':<12} {pre['mae']:>10.4f} {post['mae']:>10.4f} "
      f"  {delta_str(post['mae'], pre['mae'], higher_better=False)}")
print(f"  {'QWK':<12} {pre['qwk']:>10.4f} {post['qwk']:>10.4f} "
      f"  {delta_str(post['qwk'], pre['qwk'])}")

print(f"\n  Score distribution:")
print(f"  {'Score':<8} {'GT':>6} {'Pre':>8} {'Post':>8}")
print(f"  {'-'*32}")
for s in range(5):
    gt_c   = post["gt_distribution"].get(s, 0)
    pre_c  = pre["pred_distribution"].get(str(s), pre["pred_distribution"].get(s, 0))
    post_c = post["pred_distribution"].get(s, 0)
    print(f"  {s:<8} {gt_c:>6} {pre_c:>8} {post_c:>8}")

print(f"\n  Per-score accuracy (post fine-tuning):")
print(f"  {'Score':<8} {'Correct':>8} {'Total':>8} {'Acc':>8}")
print(f"  {'-'*36}")
for s in range(5):
    ps = post["per_score_accuracy"].get(str(s), {})
    if ps.get("total", 0) > 0:
        print(f"  {s:<8} {ps['correct']:>8} {ps['total']:>8} {ps['acc']:>8.4f}")

print(f"\n  Parse failures — Before: {pre['parse_failures']}  "
      f"After: {post['parse_failures']}")
print(f"  Inference time — {post['inference_seconds']}s total  "
      f"({post['inference_seconds']/post['n_total']:.1f}s per example)")

# ─── 9. Save all outputs ─────────────────────────────────────────────────────
paths = {
    "post_finetuning_predictions.json": results,
    "post_finetuning_metrics.json":     post,
    "comparison_report.json":           comparison,
}
for filename, data in paths.items():
    p = Path(args.out_dir) / filename
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved → {p}")

print("\nAll done. You now have everything needed for Step 6 (the report).")
print("Key files for your report:")
print("  outputs/pre_finetuning_metrics.json")
print("  outputs/post_finetuning_metrics.json")
print("  outputs/comparison_report.json")
