import json
import argparse
import os
import time
from pathlib import Path
from transformers import EarlyStoppingCallback


# 1. Define the base model configuration and maximum sequence length
# used during fine-tuning and training sample preparation
BASE_MODEL     = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
MAX_SEQ_LENGTH = 1024 # maximum token length for each training sample

# 2. Define the LoRA adaptation configuration used to efficiently fine-tune
# the base model by updating a small subset of transformer parameters
LORA_RANK      = 16    # Rank of the low-rank adaptation matrices that determine the learning capacity of the LoRA adapters
LORA_ALPHA     = 128   # Scaling factor applied to LoRA updates to control their influence during training
LORA_DROPOUT   = 0.05  # Dropout rate applied to LoRA layers to reduce overfitting and improve generalization
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # feed-forward
]

# 3. Define the training hyperparameters and output configuration used
# for LoRA fine-tuning on the Elasticsearch evaluation dataset
DEFAULT_EPOCHS       = 10     # more epochs compensate for small dataset size
DEFAULT_LR           = 2e-4   # standard for LoRA fine-tuning
DEFAULT_BATCH_SIZE   = 2      # safe for A100 40GB with 4-bit + seq len 2048
DEFAULT_GRAD_ACCUM   = 4      # A100 has enough VRAM — no accumulation needed
DEFAULT_WARMUP_STEPS = 10     # ~7% of total steps for 140 examples × 10 epochs
DEFAULT_LR_SCHEDULER = "cosine"   # Gradually decreases the learning rate following a cosine curve to improve convergence and training stability
DEFAULT_WEIGHT_DECAY = 0.01       # Applies L2 regularization to model weights to reduce overfitting and improve generalization

OUTPUT_DIR = Path("outputs/finetuned_model")

# 4. Configure command-line arguments for dataset paths, training settings,
# LoRA parameters, and model output management
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

# 5. Load the training and validation datasets and report the number
# of examples available for the fine-tuning process
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

print("Loading datasets …")
train_raw = load_jsonl(args.train_path)
val_raw   = load_jsonl(args.val_path)
print(f"  Train : {len(train_raw)} examples")
print(f"  Val   : {len(val_raw)} examples")



# 6. Load the quantized base model, configure GPU memory settings, and apply
# LoRA adapters to enable efficient parameter-efficient fine-tuning
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

# 7. Convert the formatted JSONL records into HuggingFace Dataset objects
# for efficient training and validation during fine-tuning
from datasets import Dataset

def to_hf_dataset(records):
    """Convert list of jsonl records to a HuggingFace Dataset with a 'text' column."""
    return Dataset.from_dict({"text": [r["text"] for r in records]})

train_ds = to_hf_dataset(train_raw)
val_ds   = to_hf_dataset(val_raw)

# 8. Configure the fine-tuning strategy, optimization schedule, evaluation
# settings, and checkpoint management for supervised instruction training
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
    output_dir                  = str(Path(args.out_dir) / "checkpoints"),  # Directory where checkpoints and training outputs are saved
    num_train_epochs            = args.epochs,                              # Number of complete passes over the training dataset
    per_device_train_batch_size = args.batch,                               # Number of training samples processed per GPU in each step
    per_device_eval_batch_size  = args.batch,                               # Number of validation samples processed per GPU during evaluation
    gradient_accumulation_steps = DEFAULT_GRAD_ACCUM,                       # Accumulate gradients across multiple steps to simulate a larger effective batch size
    warmup_steps                = DEFAULT_WARMUP_STEPS,                     # Gradually increase the learning rate during the initial training steps
    learning_rate               = args.lr,                                  # Initial learning rate used by the optimizer
    weight_decay                = DEFAULT_WEIGHT_DECAY,                     # L2 regularization factor to reduce overfitting
    lr_scheduler_type           = DEFAULT_LR_SCHEDULER,                     # Strategy used to adjust the learning rate throughout training

    # A100 uses bfloat16 — more stable than float16 for fine-tuning
    fp16                        = not torch.cuda.is_bf16_supported(),       # Enable float16 precision when bfloat16 is unavailable
    bf16                        = torch.cuda.is_bf16_supported(),           # Enable bfloat16 precision on supported GPUs such as A100

    logging_steps               = 5,                                        # Log training metrics every 5 optimization steps

    eval_strategy               = "steps",                                  # Perform evaluation periodically based on training steps
    eval_steps                  = 20,                                       # Run validation every 20 training steps

    save_strategy               = "steps",                                  # Save model checkpoints periodically based on training steps
    save_steps                  = 20,                                       # Save a checkpoint every 20 training steps

    load_best_model_at_end      = True,                                     # Restore the checkpoint with the best validation performance after training
    metric_for_best_model       = "eval_loss",                              # Metric used to determine the best checkpoint
    greater_is_better           = False,                                    # Lower validation loss indicates better performance

    report_to                   = "none",                                   # Disable external experiment tracking services
    seed                        = 42,                                       # Fixed random seed for reproducibility

    # Multi-GPU: HuggingFace Accelerate handles device placement automatically
    # No extra config needed — it detects all available GPUs on the node
    ddp_find_unused_parameters  = False,                                    # Disable unused parameter detection to improve distributed training efficiency
)

# 9. Initialize the supervised fine-tuning trainer with the model, datasets,
# training configuration, and early stopping mechanism
trainer = SFTTrainer(
    model              = model,              # The LoRA-enhanced Mistral model that will be fine-tuned
    tokenizer          = tokenizer,          # Tokenizer used to convert text into token IDs for the model

    train_dataset      = train_ds,           # Training dataset used to update model parameters
    eval_dataset       = val_ds,             # Validation dataset used to monitor performance during training

    dataset_text_field = "text",             # Column containing the formatted prompt-response examples
    max_seq_length     = MAX_SEQ_LENGTH,     # Maximum sequence length allowed for each training example

    dataset_num_proc   = 4,                  # Number of parallel processes used for dataset tokenization
    packing            = False,              # Keep each example separate instead of combining multiple examples into one sequence

    args               = training_args,      # Training configuration including epochs, learning rate, batch size, and evaluation settings

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=3,       # Stop training if validation loss does not improve for 3 consecutive evaluations
            early_stopping_threshold=0.001   # Minimum improvement required to be considered a meaningful gain
        )
    ]
)

# 10. Execute the fine-tuning process and report training duration,
# optimization progress, and final training performance metrics
print("\n" + "="*55)
print("Starting fine-tuning …")
print("="*55)
t0 = time.time()

trainer_stats = trainer.train()

elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed/60:.1f} minutes")
print(f"  Total steps      : {trainer_stats.global_step}")
print(f"  Final train loss : {trainer_stats.training_loss:.4f}")


# 11. Save the trained LoRA adapter and tokenizer artifacts for later
# inference, evaluation, or deployment without modifying the base model
adapter_path = Path(args.out_dir) / "lora_adapter"
print(f"\nSaving LoRA adapter → {adapter_path}")
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print("  Adapter saved.")

# 12. Optionally merge the LoRA adapter into the base model weights and save
# a standalone full-precision model for deployment and production use
if args.save_merged:
    merged_path = Path(args.out_dir) / "merged_16bit"
    print(f"\nSaving merged 16-bit model -> {merged_path}")
    print("  (this merges LoRA into base weights — takes ~3 minutes)")
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method = "merged_16bit",
    )
    print("  Merged model saved.")

# 13. Display a final fine-tuning summary that reports the training
# configuration, resource usage, model artifacts, and next evaluation step
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
print(f"\n  LoRA adapter     -> {adapter_path}")
if args.save_merged:
    print(f"  Merged model     -> {Path(args.out_dir) / 'merged_16bit'}")
