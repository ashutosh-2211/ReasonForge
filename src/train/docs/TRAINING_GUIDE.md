# LoRA Fine-tuning Training Guide

## Overview

Complete guide for LoRA fine-tuning on AWS g5.xlarge (NVIDIA A10G) using Unsloth for optimized training. This guide covers the entire pipeline from data preparation to model deployment.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Hardware Specifications](#hardware-specifications)
3. [Complete Pipeline](#complete-pipeline)
4. [Training Process](#training-process)
5. [Understanding Metrics](#understanding-metrics)
6. [Evaluation](#evaluation)
7. [Testing and Inference](#testing-and-inference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Environment Setup

### Hardware

- **Instance**: AWS g5.xlarge
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Region**: us-east-1
- **Storage**: S3 buckets
  - `reasonforge-datasets-dev-071909720457` (datasets)
  - `reasonforge-adapters-dev-071909720457` (trained adapters)
  - `reasonforge-models-dev-071909720457` (full models)

### Software Installation

```bash
# Navigate to project directory
cd /home/ec2-user/ReasonForge/src/train

# Install dependencies with uv
uv pip install -e .

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output:**
```
CUDA available: True
GPU: NVIDIA A10G
```

### AWS Configuration

Ensure your EC2 instance has IAM role with S3 permissions:

```bash
# Test S3 access
aws s3 ls s3://reasonforge-datasets-dev-071909720457/
aws s3 ls s3://reasonforge-adapters-dev-071909720457/
```

## Hardware Specifications

### NVIDIA A10G Specifications

| Specification | Value |
|--------------|-------|
| VRAM | 24 GB GDDR6 |
| CUDA Cores | 9,216 |
| Tensor Cores | 288 (3rd gen) |
| Memory Bandwidth | 600 GB/s |
| FP16 Performance | 125 TFLOPS |

### Memory Considerations

**Model Sizes:**
- Qwen3-4B Base Model: ~8 GB (4-bit quantized)
- LoRA Adapters: ~50-200 MB (depending on rank)
- Training overhead: ~4-8 GB
- Dataset in memory: ~2-4 GB

**Total Usage**: ~12-20 GB (fits comfortably in 24GB)

## Complete Pipeline

### Step 1: Data Preparation

Prepare and upload dataset to S3.

```bash
python data_preparation.py \
  --dataset "TuringEnterprises/Turing-Open-Reasoning" \
  --s3-bucket "reasonforge-datasets-dev-071909720457" \
  --val-size 0.1
```

**Output:**
```
S3 URI: s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022
```

**What happens:**
1. Downloads dataset from HuggingFace
2. Formats into Qwen3 chat template
3. Creates 90/10 train/validation split
4. Saves locally and uploads to S3

**See**: [DATA_PREPARATION.md](DATA_PREPARATION.md) for detailed guide.

### Step 2: Training

Fine-tune the model with LoRA.

```bash
python train_lora.py \
  --model-name "unsloth/Qwen2.5-Coder-3B-Instruct" \
  --dataset-path "s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022" \
  --from-s3 \
  --output-dir "./checkpoints/qwen3_4b_reasoning" \
  --num-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 16 \
  --s3-bucket "reasonforge-adapters-dev-071909720457"
```

**What happens:**
1. Downloads dataset from S3
2. Loads base model with 4-bit quantization
3. Adds LoRA adapters
4. Trains for 3 epochs
5. Saves checkpoints every 100 steps
6. Uploads final adapter to S3

**Training time**: ~2-4 hours for 50k examples (3 epochs)

### Step 3: Evaluation

Evaluate the trained model.

```bash
python evaluate_model.py \
  --model-path "./checkpoints/qwen3_4b_reasoning/final_model" \
  --is-adapter \
  --base-model "unsloth/Qwen2.5-Coder-3B-Instruct" \
  --dataset-path "s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022" \
  --split "validation" \
  --compute-perplexity \
  --generate-samples \
  --num-samples 10 \
  --output-file "./eval_results/evaluation.json"
```

**What happens:**
1. Loads base model + adapter
2. Computes perplexity on validation set
3. Generates sample outputs
4. Saves results to JSON

### Step 4: Testing

Interactive testing or batch inference.

```bash
# Interactive mode
python test_model.py \
  --model-path "./checkpoints/qwen3_4b_reasoning/final_model" \
  --is-adapter \
  --base-model "unsloth/Qwen2.5-Coder-3B-Instruct" \
  --interactive

# Single instruction test
python test_model.py \
  --model-path "./checkpoints/qwen3_4b_reasoning/final_model" \
  --is-adapter \
  --base-model "unsloth/Qwen2.5-Coder-3B-Instruct" \
  --instruction "Explain the concept of gradient descent in simple terms"
```

### Step 5: Push to HuggingFace (Optional)

```bash
# Set HuggingFace token
export HF_TOKEN="your_hf_token_here"

# Re-run training with push-to-hub flag
python train_lora.py \
  --model-name "unsloth/Qwen2.5-Coder-3B-Instruct" \
  --dataset-path "s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022" \
  --from-s3 \
  --output-dir "./checkpoints/qwen3_4b_reasoning" \
  --push-to-hub "your-username/qwen3-4b-reasoning-lora" \
  --skip-s3-upload
```

## Training Process

### LoRA Configuration

**What is LoRA?**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:
- Adds small trainable matrices to model layers
- Keeps the base model frozen
- Reduces trainable parameters by 99%
- Maintains comparable performance to full fine-tuning

**Key Parameters:**

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `lora_r` | 16 | LoRA rank | Higher = more parameters, better quality |
| `lora_alpha` | 16 | Scaling factor | Controls adapter influence |
| `lora_dropout` | 0.0 | Dropout rate | Regularization (0 = no dropout) |

**Target Modules:**
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- `gate_proj`, `up_proj`, `down_proj` (MLP layers)

**Trainable Parameters:**

For Qwen3-4B with rank=16:
- Total parameters: ~4 billion
- Trainable parameters: ~8 million
- Trainable percentage: ~0.2%

### Training Arguments

#### Batch Size and Gradient Accumulation

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
# Effective batch size = 2 × 4 = 8
```

**Why this matters:**
- Larger effective batch size = more stable gradients
- Gradient accumulation allows larger batches without OOM
- A10G with 24GB can handle batch_size=2 comfortably

**Recommendation for A10G:**
- Batch size: 1-4 (depending on sequence length)
- Gradient accumulation: 4-8
- Effective batch size: 8-16

#### Learning Rate

```python
learning_rate = 2e-4  # 0.0002
```

**Typical ranges:**
- Conservative: 1e-4 to 5e-5
- Standard: 2e-4 to 3e-4
- Aggressive: 5e-4 to 1e-3

**LoRA specific:**
- LoRA can handle higher LR than full fine-tuning
- 2e-4 is a good starting point
- Use warmup to stabilize early training

#### Warmup Steps

```python
warmup_steps = 5
```

**What it does:**
- Linearly increases LR from 0 to target over N steps
- Prevents large gradient updates early in training
- Stabilizes training

**Recommendation:**
- 5-10 steps for small datasets
- 100-500 steps for large datasets

#### Optimizer

```python
optim = "adamw_8bit"
```

**Options:**
- `adamw_8bit`: 8-bit AdamW (memory efficient)
- `adamw_torch`: Standard AdamW
- `sgd`: Simple SGD (not recommended)

**Why 8-bit?**
- Reduces optimizer memory by ~50%
- Minimal performance impact
- Allows larger batch sizes

### Training Output

```
======================================================================
🚀 LORA FINE-TUNING SETUP
======================================================================

📦 Model: unsloth/Qwen2.5-Coder-3B-Instruct

📊 Dataset:
  - Training samples: 45,000
  - Validation samples: 5,000

⚙️  Training Configuration:
  - model_name: unsloth/Qwen2.5-Coder-3B-Instruct
  - max_seq_length: 2048
  - num_train_epochs: 3
  - per_device_train_batch_size: 2
  - gradient_accumulation_steps: 4
  - learning_rate: 0.0002

🎮 GPU: NVIDIA A10G
  - Memory: 0.23GB allocated
======================================================================

🔄 Loading model: unsloth/Qwen2.5-Coder-3B-Instruct
🎮 GPU Memory - Allocated: 0.23GB | Reserved: 0.50GB
✓ Base model loaded
🎮 GPU Memory - Allocated: 7.82GB | Reserved: 8.00GB

🔧 Adding LoRA adapters...
✓ LoRA adapters added
🎮 GPU Memory - Allocated: 7.83GB | Reserved: 8.02GB

📊 Model Parameters:
  - Total: 3,758,096,384
  - Trainable: 8,388,608
  - Trainable %: 0.22%

🏋️  Initializing SFTTrainer...
✓ Trainer initialized

🚀 Starting training...
======================================================================
[1/16875] Loss: 1.8234, LR: 0.000040
[100/16875] Loss: 1.2341, LR: 0.000200
[200/16875] Loss: 1.1256, LR: 0.000200
...
[16800/16875] Loss: 0.3421, LR: 0.000001
======================================================================
✅ Training completed in 2h 34m 18s

📊 Final Training Metrics:
  - Training Loss: 0.3421
  - Training Steps: 16875
  - Training Time: 2h 34m 18s

💾 Saving final model to checkpoints/qwen3_4b_reasoning/final_model...
✓ Model and metrics saved
```

## Understanding Metrics

### Training Loss

**What it is:**
- Measure of how well the model predicts the training data
- Lower = better fit to training data

**Typical progression:**
- Start: 2.0-3.0
- End: 0.2-0.5

**Warning signs:**
- Not decreasing: Learning rate too low or data issues
- Decreasing too fast: Learning rate too high, overfitting risk
- Fluctuating wildly: Batch size too small

### Validation Loss

**What it is:**
- Measure of model performance on unseen data
- Lower = better generalization

**Interpretation:**
```
Train Loss  | Val Loss  | Interpretation
------------|-----------|------------------
1.0         | 1.1       | Good (slight generalization gap)
0.5         | 0.6       | Excellent
0.3         | 1.2       | Overfitting (train much better than val)
1.5         | 1.4       | Underfitting (model not learning)
```

**Best model selection:**
- Automatically saves checkpoint with lowest validation loss
- Use `load_best_model_at_end=True` (default in our script)

### Perplexity (PPL)

**What it is:**
- Exponential of average loss: `PPL = exp(loss)`
- Measures how "surprised" the model is by the text
- Lower = better

**Typical values:**
```
PPL Range  | Quality
-----------|------------------
< 10       | Excellent
10-20      | Good
20-50      | Acceptable
> 50       | Poor
```

**Example:**
```
Validation Loss: 0.693
Perplexity: exp(0.693) = 2.0
```

Interpretation: On average, the model is choosing between ~2 equally likely tokens (very good).

### Learning Rate Schedule

**Linear decay** (default):
```
LR starts at 2e-4
Warmup for 5 steps: 0 → 2e-4
Decay linearly: 2e-4 → 0
```

**Why it matters:**
- High LR early: Fast learning
- Low LR late: Fine-tuning, stability

### GPU Memory Usage

**What to watch:**
```
Allocated: Memory actively used by tensors
Reserved: Memory reserved by PyTorch
```

**Example:**
```
Before loading: 0.2 GB allocated
After base model: 7.8 GB allocated
After LoRA: 8.0 GB allocated
During training: 14-18 GB allocated (peaks with gradients)
```

**A10G capacity**: 24 GB
**Safe usage**: < 20 GB

## Evaluation

### Perplexity Computation

```bash
python evaluate_model.py \
  --model-path "./checkpoints/qwen3_4b_reasoning/final_model" \
  --is-adapter \
  --dataset-path "s3://your-dataset-path" \
  --split "validation" \
  --compute-perplexity
```

**Output:**
```
📊 Computing perplexity...
  Batch 10: Current PPL = 12.45
  Batch 20: Current PPL = 10.32
  ...

✓ Perplexity Metrics:
  - Perplexity: 8.24
  - Average Loss: 2.109
  - Samples Evaluated: 5,000
```

**Interpretation:**

| Metric | Value | Meaning |
|--------|-------|---------|
| Perplexity | 8.24 | Model performs well; choosing from ~8 options on average |
| Avg Loss | 2.109 | log(8.24) - mathematically equivalent to PPL |

**Comparison:**

Compare with base model to measure improvement:
```
Base Model PPL: 25.3
Fine-tuned PPL: 8.2
Improvement: 67.6% reduction
```

### Sample Generation

Generates actual outputs to qualitatively assess performance.

```bash
python evaluate_model.py \
  --model-path "./checkpoints/qwen3_4b_reasoning/final_model" \
  --is-adapter \
  --dataset-path "s3://your-dataset-path" \
  --generate-samples \
  --num-samples 10
```

**Output:**
```
======================================================================
Sample 1:
======================================================================
Instruction: Explain the concept of gradient descent...

Ground Truth: Gradient descent is an optimization algorithm...

Generated: Gradient descent is a fundamental optimization algorithm...
```

**What to check:**
- **Coherence**: Does it make sense?
- **Relevance**: Does it answer the question?
- **Style**: Does it match the training data?
- **Accuracy**: Is the information correct?

## Testing and Inference

### Interactive Testing

```bash
python test_model.py \
  --model-path "./checkpoints/qwen3_4b_reasoning/final_model" \
  --is-adapter \
  --interactive
```

**Session:**
```
======================================================================
🎯 INTERACTIVE TESTING MODE
======================================================================
Type your questions below. Type 'quit' to stop.
Type 'settings' to change generation parameters.
======================================================================

💬 You: What is the capital of France?

🤖 Assistant: The capital of France is Paris. It is not only the largest
city in France but also its political, economic, and cultural center...

💬 You: settings

Current settings:
  - max_new_tokens: 512
  - temperature: 0.7
New max_new_tokens [512]: 256
New temperature [0.7]: 0.5
✓ Settings updated!

💬 You: quit

👋 Goodbye!
```

### Generation Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | 0.1-2.0 | 0.7 | Randomness (lower = more deterministic) |
| `top_p` | 0.1-1.0 | 0.9 | Nucleus sampling (smaller = more focused) |
| `top_k` | 1-100 | 50 | Top-k sampling (smaller = more conservative) |
| `max_new_tokens` | 1-2048 | 512 | Maximum response length |
| `repetition_penalty` | 1.0-2.0 | 1.1 | Penalty for repetition (higher = less repetition) |

**Recommendations:**

**For factual answers:**
```python
temperature = 0.3
top_p = 0.9
top_k = 40
```

**For creative writing:**
```python
temperature = 0.9
top_p = 0.95
top_k = 80
```

**For reasoning tasks:**
```python
temperature = 0.7
top_p = 0.9
top_k = 50
```

### Batch Inference

```bash
# Create instructions file
cat > instructions.json << EOF
[
  "Explain quantum computing",
  "What is machine learning?",
  "Describe photosynthesis"
]
EOF

# Run batch inference
python test_model.py \
  --model-path "./checkpoints/qwen3_4b_reasoning/final_model" \
  --is-adapter \
  --batch \
  --instructions-file instructions.json \
  --output-file results.json
```

## Best Practices

### 1. Start Small

```bash
# Test with 1000 samples first
python data_preparation.py --max-samples 1000 --skip-upload
python train_lora.py \
  --dataset-path "./data_cache/processed_reasoning" \
  --num-epochs 1 \
  --skip-s3-upload
```

### 2. Monitor Training

Watch for these signs:

**Good training:**
- Loss steadily decreasing
- Validation loss tracking training loss
- GPU memory stable
- No NaN values

**Bad training:**
- Loss not decreasing
- Validation loss increasing (overfitting)
- GPU OOM errors
- NaN or Inf losses

### 3. Checkpoint Management

```python
save_steps = 100          # Save every 100 steps
save_total_limit = 3      # Keep only last 3 checkpoints
```

**Why:**
- Recover from crashes
- Resume training
- Test different checkpoints
- Save disk space

### 4. Experiment Tracking

Keep a training log:

```bash
# Create experiment log
cat > experiment_log.md << EOF
## Experiment 1: Baseline
- Date: 2025-12-20
- Model: Qwen3-4B
- Dataset: Turing-Open-Reasoning (1000 samples)
- LoRA rank: 16
- Learning rate: 2e-4
- Epochs: 3
- Final train loss: 0.45
- Final val loss: 0.52
- Perplexity: 1.68
- Notes: Good baseline, no overfitting
EOF
```

### 5. Hyperparameter Tuning

**Priority order:**

1. **Learning rate** (biggest impact)
   - Try: 1e-4, 2e-4, 5e-4
2. **LoRA rank** (quality vs. size)
   - Try: 8, 16, 32, 64
3. **Batch size** (stability)
   - Try: 4, 8, 16 (effective)
4. **Epochs** (prevent over/underfitting)
   - Try: 1, 2, 3, 5

### 6. Gradient Checkpointing

Already enabled via Unsloth:
```python
use_gradient_checkpointing = "unsloth"
```

**Benefit:**
- Reduces memory by ~40%
- Allows larger batch sizes
- Slightly slower training

## Troubleshooting

### GPU Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce batch size:
```bash
--batch-size 1 --gradient-accumulation-steps 8
```

2. Reduce sequence length:
```bash
--max-seq-length 1024  # instead of 2048
```

3. Use gradient checkpointing (already enabled)

4. Clear CUDA cache:
```python
import torch
torch.cuda.empty_cache()
```

### Training Loss Not Decreasing

**Possible causes:**

1. **Learning rate too low**
```bash
--learning-rate 5e-4  # increase from 2e-4
```

2. **Learning rate too high** (loss oscillates)
```bash
--learning-rate 1e-4  # decrease from 2e-4
```

3. **Data formatting issues**
   - Check sample outputs
   - Verify chat template is correct

4. **Insufficient epochs**
```bash
--num-epochs 5  # increase from 3
```

### Overfitting

**Symptoms:**
- Train loss << Validation loss
- Validation loss increasing while train loss decreasing

**Solutions:**

1. **Reduce epochs**
```bash
--num-epochs 2
```

2. **Add regularization**
```bash
--weight-decay 0.1  # increase from 0.01
--lora-dropout 0.1  # add dropout
```

3. **Get more data**
   - Use full dataset instead of subset
   - Add data augmentation

4. **Reduce model capacity**
```bash
--lora-r 8  # reduce from 16
```

### S3 Upload Failures

**Symptoms:**
```
AccessDenied
NoSuchBucket
```

**Solutions:**

1. Check IAM permissions:
```bash
aws sts get-caller-identity
aws s3 ls s3://your-bucket/
```

2. Verify bucket exists:
```bash
aws s3 mb s3://your-bucket  # create if needed
```

3. Check region:
```bash
aws configure get region
```

### Model Loading Errors

**Symptoms:**
```
OSError: Can't load checkpoint
```

**Solutions:**

1. Verify path exists:
```bash
ls -la ./checkpoints/qwen3_4b_reasoning/final_model
```

2. Check for required files:
```
adapter_config.json
adapter_model.safetensors
```

3. Use correct loading method:
```bash
# For adapter
--is-adapter --base-model "unsloth/Qwen2.5-Coder-3B-Instruct"

# For full model
# (no --is-adapter flag)
```

## Performance Benchmarks

### Training Speed (A10G)

| Dataset Size | Epochs | Batch Size | Time |
|-------------|--------|------------|------|
| 1,000 samples | 3 | 8 (eff) | ~5 min |
| 10,000 samples | 3 | 8 (eff) | ~45 min |
| 50,000 samples | 3 | 8 (eff) | ~2.5 hours |
| 100,000 samples | 3 | 8 (eff) | ~5 hours |

**With Unsloth optimization:**
- 2x faster than standard training
- 50% less memory usage
- Same final performance

### Model Quality Benchmarks

**Example results** (your results may vary):

| Metric | Base Model | After Fine-tuning |
|--------|-----------|-------------------|
| Perplexity (val) | 25.3 | 8.2 |
| Reasoning Quality | Moderate | High |
| Task Accuracy | 60% | 85% |

## File Reference

### Project Structure

```
/home/ec2-user/ReasonForge/src/train/
├── data_preparation.py       # Data processing script
├── train_lora.py              # Training script
├── evaluate_model.py          # Evaluation script
├── test_model.py              # Testing/inference script
├── utils.py                   # Utility functions
├── pyproject.toml             # Dependencies
├── DATA_PREPARATION.md        # Data prep guide
├── TRAINING_GUIDE.md          # This file
├── README.md                  # Project overview
├── data_cache/                # Local dataset cache
├── checkpoints/               # Training checkpoints
├── eval_results/              # Evaluation outputs
└── test_results/              # Test outputs
```

### Generated Artifacts

**During training:**
```
checkpoints/qwen3_4b_reasoning/
├── checkpoint-100/
├── checkpoint-200/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   └── ...
├── training_config.json
├── training_metrics.json
└── logs/
```

**After evaluation:**
```
eval_results/
└── evaluation.json
```

**After testing:**
```
test_results/
└── inference.json
```

## Next Steps

After successful training:

1. **Evaluate thoroughly**: Test on multiple examples
2. **Compare with base model**: Measure actual improvement
3. **Deploy**: Push to HuggingFace or serve via API
4. **Iterate**: Adjust hyperparameters and retrain
5. **Scale up**: Train on larger datasets or for more epochs

## Additional Resources

- **Unsloth Documentation**: https://github.com/unslothai/unsloth
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Qwen Model**: https://huggingface.co/Qwen
- **AWS A10G**: https://aws.amazon.com/ec2/instance-types/g5/

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting)
2. Review error logs in `checkpoints/*/logs/`
3. Verify GPU status: `nvidia-smi`
4. Check disk space: `df -h`
