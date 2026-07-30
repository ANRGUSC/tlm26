"""
Knowledge Distillation training script.
Trains a small student model using soft labels from a larger teacher model.

Usage:
  python train_distill.py --teacher_checkpoint=out_D/ckpt.pt --out_dir=out_distill \
    --vocab_source=custom --vocab_size=512 --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 \
    --batch_size=128 --max_iters=5000 --eval_interval=500 --compile=False

Distillation params:
  --distill_alpha=0.5   weight for soft (distillation) loss vs hard (CE) loss
  --distill_temp=3.0    temperature for softening teacher logits
"""

import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
import torch.nn.functional as F
from model import Transformer, ModelArgs
from tinystories import Task
from export import model_export

# --- Extract distillation args before configurator ---
teacher_checkpoint = "out_D/ckpt.pt"
distill_alpha = 0.5  # 0.5 = equal weight hard/soft loss
distill_temp = 3.0   # temperature for softening logits

filtered_argv = []
for arg in sys.argv:
    if arg.startswith('--teacher_checkpoint='):
        teacher_checkpoint = arg.split('=', 1)[1]
    elif arg.startswith('--distill_alpha='):
        distill_alpha = float(arg.split('=')[1])
    elif arg.startswith('--distill_temp='):
        distill_temp = float(arg.split('=')[1])
    else:
        filtered_argv.append(arg)
sys.argv = filtered_argv

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = "scratch"
wandb_log = False
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 128
max_seq_len = 256
vocab_source = "llama2"
vocab_size = 32000
# model (student)
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# adamw optimizer
gradient_accumulation_steps = 4
learning_rate = 5e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 1000
# system
device = "cuda"
dtype = "bfloat16"
compile = True
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

lr_decay_iters = max_iters
min_lr = 0.0

assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000

# various inits
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    from torch.distributed import init_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# ============================================================
# Load teacher model (frozen)
# ============================================================
print("=" * 50)
print(f"  KNOWLEDGE DISTILLATION")
print(f"  Teacher: {teacher_checkpoint}")
print(f"  Alpha (soft weight): {distill_alpha}")
print(f"  Temperature: {distill_temp}")
print("=" * 50)

teacher_ckpt = torch.load(teacher_checkpoint, map_location=device)
teacher_args = ModelArgs(**teacher_ckpt["model_args"])
teacher_model = Transformer(teacher_args)
teacher_state = teacher_ckpt["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(teacher_state.items()):
    if k.startswith(unwanted_prefix):
        teacher_state[k[len(unwanted_prefix):]] = teacher_state.pop(k)
teacher_model.load_state_dict(teacher_state)
teacher_model.to(device)
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

teacher_params = sum(p.numel() for p in teacher_model.parameters())
print(f"  Teacher params: {teacher_params:,}")
print(f"  Teacher val loss: {teacher_ckpt.get('best_val_loss', 'unknown')}")
teacher_ckpt = None  # free memory

# ============================================================
# Init student model
# ============================================================
iter_num = 0
best_val_loss = 1e9

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)

print("Initializing student model from scratch")
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
student_params = sum(p.numel() for p in model.parameters())
print(f"  Student params: {student_params:,}")
print(f"  Compression ratio: {teacher_params/student_params:.1f}x")
model.to(device)

# optimizer
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the student model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

raw_model = model

# ============================================================
# Loss functions
# ============================================================
def distillation_loss(student_logits, teacher_logits, targets, alpha, temp):
    """Combined hard + soft distillation loss."""
    # Hard loss: standard cross-entropy with true labels
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
    )

    # Soft loss: KL divergence between teacher and student soft distributions
    # Scale logits by temperature before softmax
    student_soft = F.log_softmax(student_logits / temp, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
    soft_loss = F.kl_div(
        student_soft.view(-1, student_soft.size(-1)),
        teacher_soft.view(-1, teacher_soft.size(-1)),
        reduction="batchmean",
    )
    # Scale by T^2 as per Hinton et al.
    soft_loss = soft_loss * (temp ** 2)

    # Combined loss
    total = (1 - alpha) * hard_loss + alpha * soft_loss
    return total, hard_loss, soft_loss


# ============================================================
# Evaluation (uses standard CE, not distillation loss)
# ============================================================
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# Training loop
# ============================================================
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
    if iter_num == 0 and eval_only:
        break

    # forward/backward with distillation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            # pass Y to get full-sequence logits (not just last position)
            student_logits = model(X, Y)
            with torch.no_grad():
                teacher_logits = teacher_model(X, Y)

            loss, hard_loss, soft_loss = distillation_loss(
                student_logits, teacher_logits, Y, distill_alpha, distill_temp
            )
            loss = loss / gradient_accumulation_steps

        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"{iter_num} | loss {lossf:.4f} (hard={hard_loss.item():.4f} soft={soft_loss.item():.4f}) | lr {lr:.6e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")
    local_iter_num += 1
    iter_num += 1

    if iter_num > max_iters:
        break

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
