"""
04_finetune.py
===============
Step 4 — Fine-tune Mistral-7B-Instruct-v0.2 with Unsloth + LoRA.

Optimised for NVIDIA A100 (40 GB or 80 GB).
Automatically uses all GPUs visible on the current node via HuggingFace
Accelerate — no SLURM or torchrun config required.

Reads  : formatted/train.jsonl
         formatted/val.jsonl
Writes : outputs/finetuned_model/lora_adapter/   ← LoRA weights + tokenizer
         outputs/finetuned_model/merged_16bit/    ← optional full merged model

Single-GPU run:
    python 04_finetune.py

Multi-GPU run (uses all GPUs on the node automatically):
    accelerate launch --multi_gpu 04_finetune.py

With custom args:
    python 04_finetune.py --epochs 10 --batch 8 --lr 2e-4
"""

import json
import argparse
import os
import time
from pathlib import Path
from transformers import EarlyStoppingCallback
# ─── Defaults ────────────────────────────────────────────────────────────────
BASE_MODEL     = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
MAX_SEQ_LENGTH = 1024 # maximum token length for each training sample

# LoRA config — these values are well-tested for instruction tuning
# on small datasets with Mistral-7B
LORA_RANK      = 16    # higher rank = more capacity; 64 is a good balance for A100
LORA_ALPHA     = 128   # typically 2× rank
LORA_DROPOUT   = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # feed-forward
]

# Training defaults — tuned for A100 + small dataset (140 examples)
DEFAULT_EPOCHS       = 10     # more epochs compensate for small dataset size
DEFAULT_LR           = 2e-4   # standard for LoRA fine-tuning
DEFAULT_BATCH_SIZE   = 2      # safe for A100 40GB with 4-bit + seq len 2048
DEFAULT_GRAD_ACCUM   = 4      # A100 has enough VRAM — no accumulation needed
DEFAULT_WARMUP_STEPS = 10     # ~7% of total steps for 140 examples × 10 epochs
DEFAULT_LR_SCHEDULER = "cosine"
DEFAULT_WEIGHT_DECAY = 0.01

OUTPUT_DIR = Path("outputs/finetuned_model")

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train_path", default="jsonl_format/train.jsonl")
parser.add_argument("--val_path",   default="jsonl_format/val.jsonl")
parser.add_argument("--out_dir",    default=str(OUTPUT_DIR))
parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
parser.add_argument("--batch",      type=int,   default=DEFAULT_BATCH_SIZE)
parser.add_argument("--lora_rank",  type=int,   default=LORA_RANK)
parser.add_argument("--save_merged", action="store_true",
                    help="Also save the full merged 16-bit model (large, ~14 GB)")
args = parser.parse_args()

Path(args.out_dir).mkdir(parents=True, exist_ok=True)

# ─── 1. Load datasets ────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

print("Loading datasets …")
train_raw = load_jsonl(args.train_path)
val_raw   = load_jsonl(args.val_path)
print(f"  Train : {len(train_raw)} examples")
print(f"  Val   : {len(val_raw)} examples")



# ─── 2. Load base model + apply LoRA ─────────────────────────────────────────
print(f"\nLoading base model: {BASE_MODEL}")
print("  (downloads ~4 GB of weights on first run)")

from unsloth import FastLanguageModel
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,       # auto: bfloat16 on A100, float16 on older GPUs
    load_in_4bit   = True,       # 4-bit base keeps VRAM low; LoRA adapters are full precision
)

# Detect and log precision
dtype_used = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
print(f"  Precision      : {dtype_used} (A100 supports bfloat16 — more numerically stable)")
print(f"  GPU count      : {torch.cuda.device_count()}")
if torch.cuda.device_count() > 1:
    print(f"  Multi-GPU mode : enabled (using all {torch.cuda.device_count()} GPUs on this node)")

# Apply LoRA adapters
print(f"\nApplying LoRA (rank={args.lora_rank}, alpha={LORA_ALPHA}) …")
model = FastLanguageModel.get_peft_model(
    model,
    r                   = args.lora_rank,
    target_modules      = LORA_TARGET_MODULES,
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = LORA_DROPOUT,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth's optimised checkpointing
    random_state        = 42,
    use_rslora          = False,
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.2f}% of {total:,} total)")

# ─── 3. Prepare HuggingFace datasets ─────────────────────────────────────────
from datasets import Dataset

def to_hf_dataset(records):
    """Convert list of jsonl records to a HuggingFace Dataset with a 'text' column."""
    return Dataset.from_dict({"text": [r["text"] for r in records]})

train_ds = to_hf_dataset(train_raw)
val_ds   = to_hf_dataset(val_raw)

# ─── 4. Training arguments ───────────────────────────────────────────────────
from trl import SFTTrainer
from transformers import TrainingArguments

# Total steps = (train_size / batch_size) * epochs
# With 140 examples, batch=8, epochs=10 → ~175 steps total
total_steps = (len(train_raw) // args.batch) * args.epochs
print(f"\nTraining plan:")
print(f"  Examples   : {len(train_raw)}")
print(f"  Batch size : {args.batch}  ×  grad_accum={DEFAULT_GRAD_ACCUM}")
print(f"  Epochs     : {args.epochs}")
print(f"  Total steps: ~{total_steps}")
print(f"  LR         : {args.lr}  scheduler={DEFAULT_LR_SCHEDULER}")

training_args = TrainingArguments(
    output_dir                  = str(Path(args.out_dir) / "checkpoints"),
    num_train_epochs            = args.epochs,
    per_device_train_batch_size = args.batch,
    per_device_eval_batch_size  = args.batch,
    gradient_accumulation_steps = DEFAULT_GRAD_ACCUM,
    warmup_steps                = DEFAULT_WARMUP_STEPS,
    learning_rate               = args.lr,
    weight_decay                = DEFAULT_WEIGHT_DECAY,
    lr_scheduler_type           = DEFAULT_LR_SCHEDULER,
    # A100 uses bfloat16 — more stable than float16 for fine-tuning
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 5,
    eval_strategy         = "steps",
    eval_steps                  = 20,
    save_strategy               = "steps",
    save_steps                  = 20,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    report_to                   = "none",   # set to "wandb" if you use Weights & Biases
    seed                        = 42,
    # Multi-GPU: HuggingFace Accelerate handles device placement automatically
    # No extra config needed — it detects all available GPUs on the node
    ddp_find_unused_parameters  = False,    # speeds up DDP; safe for LoRA
)

# ─── 5. SFTTrainer ───────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    dataset_num_proc   = 4,       # parallel tokenisation workers; safe on A100 nodes
    packing            = False,   # keep off for instruction tuning — don't mix examples
    args               = training_args,
    callbacks = [
     EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
] 
)

# ─── 6. Train ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Starting fine-tuning …")
print("="*55)
t0 = time.time()

trainer_stats = trainer.train()

elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed/60:.1f} minutes")
print(f"  Total steps      : {trainer_stats.global_step}")
print(f"  Final train loss : {trainer_stats.training_loss:.4f}")

# ─── 7. Save LoRA adapter ────────────────────────────────────────────────────
adapter_path = Path(args.out_dir) / "lora_adapter"
print(f"\nSaving LoRA adapter → {adapter_path}")
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print("  Adapter saved.")

# ─── 8. (Optional) Save merged 16-bit model ──────────────────────────────────
# The merged model bakes the LoRA weights back into the base weights.
# Useful for deployment but large (~14 GB). Skip unless you need it.
if args.save_merged:
    merged_path = Path(args.out_dir) / "merged_16bit"
    print(f"\nSaving merged 16-bit model → {merged_path}")
    print("  (this merges LoRA into base weights — takes ~3 minutes)")
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method = "merged_16bit",
    )
    print("  Merged model saved.")

# ─── 9. Training summary ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINE-TUNING SUMMARY")
print("="*55)
print(f"  Base model       : {BASE_MODEL}")
print(f"  LoRA rank        : {args.lora_rank}  alpha={LORA_ALPHA}")
print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.2f}%)")
print(f"  Epochs           : {args.epochs}")
print(f"  Batch size       : {args.batch}")
print(f"  Learning rate    : {args.lr}")
print(f"  Precision        : {dtype_used}")
print(f"  GPU count        : {torch.cuda.device_count()}")
print(f"  Total time       : {elapsed/60:.1f} minutes")
print(f"  Final train loss : {trainer_stats.training_loss:.4f}")
print(f"\n  LoRA adapter     → {adapter_path}")
if args.save_merged:
    print(f"  Merged model     → {Path(args.out_dir) / 'merged_16bit'}")
print(f"\nNext step: run 05_post_finetuning_eval.py --adapter {adapter_path}")
