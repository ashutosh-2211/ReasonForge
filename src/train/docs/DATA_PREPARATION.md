# Data Preparation Guide

## Overview

This guide explains the data preparation pipeline for LoRA fine-tuning. The `data_preparation.py` script handles downloading datasets from HuggingFace, formatting them for training, and uploading to S3.

## Architecture

```
HuggingFace Hub → Download → Format → Split → Save Locally → Upload to S3
```

## Dataset Information

- **Source**: TuringEnterprises/Turing-Open-Reasoning
- **Type**: Reasoning and explanation dataset
- **Format**: Instruction-response pairs with detailed reasoning
- **Storage**: AWS S3 (reasonforge-datasets-dev-071909720457)

## Script Components

### 1. DatasetProcessor Class

The main class that handles all data operations.

#### Initialization

```python
processor = DatasetProcessor(
    dataset_name="TuringEnterprises/Turing-Open-Reasoning",
    s3_bucket="reasonforge-datasets-dev-071909720457",
    local_cache_dir="./data_cache",
    aws_region="us-east-1"
)
```

**Parameters:**
- `dataset_name`: HuggingFace dataset identifier
- `s3_bucket`: S3 bucket for storage
- `local_cache_dir`: Local directory for caching
- `aws_region`: AWS region for S3 operations

### 2. Loading Dataset from HuggingFace

**Function**: `load_dataset_from_hf()`

Downloads the dataset from HuggingFace Hub.

```python
dataset = processor.load_dataset_from_hf(split=None)
```

**Parameters:**
- `split`: Specific split to load ('train', 'test', etc.). None loads all splits.

**Returns:**
- Dataset or DatasetDict object

**What it does:**
- Downloads dataset from HuggingFace
- Caches locally for faster subsequent loads
- Reports dataset statistics (number of examples per split)

### 3. Formatting for Training

**Function**: `format_dataset_for_training()`

Formats raw dataset into instruction-following format compatible with Qwen3.

```python
formatted_dataset = processor.format_dataset_for_training(dataset)
```

**Chat Format Template:**

```
<|im_start|>system
You are a helpful assistant that provides detailed reasoning and explanations.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

**Field Mapping:**

The script intelligently detects dataset structure and maps fields:

| Dataset Fields | Mapped To |
|---------------|-----------|
| question + reasoning | instruction + response |
| prompt + completion | instruction + response |
| input + output | instruction + response |

**Output Fields:**
- `text`: Complete formatted conversation (used for training)
- `instruction`: Original question/prompt
- `response`: Original answer/completion

### 4. Creating Train/Validation Split

**Function**: `create_train_val_split()`

Splits dataset into training and validation sets.

```python
dataset_dict = processor.create_train_val_split(
    dataset=formatted_dataset,
    val_size=0.1,  # 10% for validation
    seed=42
)
```

**Parameters:**
- `dataset`: Formatted dataset to split
- `val_size`: Fraction for validation (0.0-1.0)
- `seed`: Random seed for reproducibility

**Returns:**
- DatasetDict with 'train' and 'validation' splits

**Best Practices:**
- Use 10-20% for validation
- Always set a seed for reproducibility
- Ensure validation set is large enough (min 100-500 examples)

### 5. Saving Locally

**Function**: `save_dataset_locally()`

Saves processed dataset to local disk.

```python
local_path = processor.save_dataset_locally(
    dataset=dataset_dict,
    name="processed_reasoning"
)
```

**What it saves:**
- All dataset splits
- Schema information
- Dataset metadata
- Arrow format files for fast loading

**Directory Structure:**
```
data_cache/
└── processed_reasoning/
    ├── train/
    │   ├── data-00000-of-00001.arrow
    │   └── state.json
    ├── validation/
    │   ├── data-00000-of-00001.arrow
    │   └── state.json
    └── dataset_dict.json
```

### 6. Uploading to S3

**Function**: `upload_to_s3()`

Uploads processed dataset to S3 for distributed access.

```python
s3_uri = processor.upload_to_s3(
    local_path=local_path,
    s3_prefix="datasets"
)
```

**S3 Path Structure:**
```
s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022/
├── train/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── validation/
│   ├── data-00000-of-00001.arrow
│   └── state.json
└── dataset_dict.json
```

**Metadata:**
- Saves upload metadata JSON with timestamp
- Includes list of all uploaded files
- Records S3 URI for easy reference

### 7. Downloading from S3

**Function**: `download_from_s3()`

Downloads dataset from S3 to local directory.

```python
local_path = processor.download_from_s3(
    s3_uri="s3://bucket/prefix",
    local_path=Path("./data_cache/downloaded")
)
```

**Use case:**
- Loading datasets on different machines
- Sharing processed datasets across team
- Training on cloud instances

## Usage Examples

### Basic Usage

```bash
# Prepare dataset with default settings
python data_preparation.py

# Outputs:
# - Local: ./data_cache/processed_reasoning/
# - S3: s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_TIMESTAMP/
```

### Custom Dataset

```bash
python data_preparation.py \
  --dataset "username/my-dataset" \
  --s3-bucket "my-bucket"
```

### Testing with Limited Samples

```bash
# Process only 1000 samples for quick testing
python data_preparation.py \
  --max-samples 1000 \
  --skip-upload
```

### Custom Validation Split

```bash
# Use 20% for validation instead of default 10%
python data_preparation.py \
  --val-size 0.2
```

### Skip S3 Upload (Local Only)

```bash
python data_preparation.py \
  --skip-upload
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | TuringEnterprises/Turing-Open-Reasoning | HuggingFace dataset name |
| `--s3-bucket` | str | reasonforge-datasets-dev-071909720457 | S3 bucket name |
| `--val-size` | float | 0.1 | Validation split size (0.0-1.0) |
| `--max-samples` | int | None | Max samples to process (for testing) |
| `--skip-upload` | flag | False | Skip S3 upload |

## Understanding the Output

### Console Output

```
======================================================================
DATA PREPARATION PIPELINE
======================================================================
✓ Initialized DatasetProcessor
  Dataset: TuringEnterprises/Turing-Open-Reasoning
  S3 Bucket: reasonforge-datasets-dev-071909720457
  Local Cache: ./data_cache

📥 Loading dataset from HuggingFace Hub...
✓ Loaded dataset with splits: ['train']
  - train: 50,000 examples

🔄 Formatting dataset for training...
✓ Formatted 50,000 examples
  Sample formatted text (first 200 chars):
  <|im_start|>system
You are a helpful assistant that provides detailed reasoning...

✂️  Creating train/validation split (val_size=0.1)...
✓ Split created:
  - Training: 45,000 examples
  - Validation: 5,000 examples

💾 Saving dataset locally to data_cache/processed_reasoning...
✓ Dataset saved successfully

☁️  Uploading dataset to S3...
  Bucket: s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022
  ✓ Uploaded: train/data-00000-of-00001.arrow
  ✓ Uploaded: train/state.json
  ✓ Uploaded: validation/data-00000-of-00001.arrow
  ✓ Uploaded: validation/state.json
  ✓ Uploaded: dataset_dict.json

✓ Upload complete! Total files: 5
  S3 URI: s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022
  Metadata saved to: data_cache/upload_metadata_20251220_143022.json

======================================================================
✅ DATA PREPARATION COMPLETE!
======================================================================
Local Path: data_cache/processed_reasoning
S3 URI: s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022

To use this dataset in training, set:
  --dataset-path 's3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022'
======================================================================
```

### Metadata File

The upload generates a metadata JSON file:

```json
{
  "dataset_name": "TuringEnterprises/Turing-Open-Reasoning",
  "s3_uri": "s3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_20251220_143022",
  "timestamp": "20251220_143022",
  "files": [
    "datasets/processed_reasoning_20251220_143022/train/data-00000-of-00001.arrow",
    "datasets/processed_reasoning_20251220_143022/train/state.json",
    ...
  ]
}
```

## Dataset Format Details

### Input Dataset Structure

Expected fields in source dataset (in order of preference):

1. **Primary**: `question` + `reasoning`
2. **Alternative 1**: `prompt` + `completion`
3. **Alternative 2**: `input` + `output`
4. **Fallback**: First two string fields

### Output Dataset Structure

After processing, each example has:

```python
{
    "text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>",
    "instruction": "Original question or prompt",
    "response": "Original answer or completion"
}
```

**Field Purposes:**
- `text`: Used directly for training (contains full formatted conversation)
- `instruction`: For reference and evaluation
- `response`: For reference and evaluation

## AWS Configuration

### Required IAM Permissions

Your EC2 instance or IAM role needs:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::reasonforge-datasets-dev-071909720457",
                "arn:aws:s3:::reasonforge-datasets-dev-071909720457/*"
            ]
        }
    ]
}
```

### S3 Bucket Structure

```
reasonforge-datasets-dev-071909720457/
└── datasets/
    ├── processed_reasoning_20251220_143022/
    ├── processed_reasoning_20251220_150000/
    └── ...
```

## Troubleshooting

### Common Issues

#### 1. HuggingFace Download Fails

**Error**: `Connection timeout` or `Repository not found`

**Solution:**
```bash
# Ensure HuggingFace token is set if dataset is private
export HF_TOKEN="your_token_here"

# Or use huggingface-cli login
huggingface-cli login
```

#### 2. S3 Upload Fails

**Error**: `AccessDenied` or `NoSuchBucket`

**Solution:**
- Check IAM permissions
- Verify bucket exists and region is correct
- Test AWS credentials: `aws s3 ls s3://your-bucket/`

#### 3. Out of Memory

**Error**: `RuntimeError: CUDA out of memory` or system hangs

**Solution:**
```bash
# Process fewer samples at a time
python data_preparation.py --max-samples 10000

# Or use a machine with more RAM
```

#### 4. Formatting Issues

**Error**: Dataset fields don't match expected format

**Solution:**
- Check your dataset structure on HuggingFace
- Modify the `format_example()` function in data_preparation.py:243-145
- Add custom field mapping for your dataset

## Best Practices

### 1. Data Quality

- **Inspect samples**: Always check a few formatted examples
- **Verify splits**: Ensure train/val split is reasonable
- **Check balance**: Validate that splits have similar distributions

### 2. Storage Management

- **Version datasets**: Keep timestamp in S3 paths for versioning
- **Clean old datasets**: Remove outdated datasets to save storage costs
- **Use metadata**: Always save and reference metadata files

### 3. Reproducibility

- **Fixed seeds**: Always use the same seed for splitting
- **Document versions**: Record dataset versions and timestamps
- **Save configs**: Keep track of processing parameters

### 4. Testing

- **Start small**: Test with `--max-samples 100` first
- **Validate format**: Check that the chat template is correct
- **Skip upload**: Use `--skip-upload` during testing

## Next Steps

After preparing your dataset:

1. **Note the S3 URI** from the output
2. **Use it in training**:
   ```bash
   python train_lora.py \
     --dataset-path 's3://reasonforge-datasets-dev-071909720457/datasets/processed_reasoning_TIMESTAMP' \
     --from-s3
   ```
3. **See TRAINING_GUIDE.md** for complete training instructions

## Reference

### Key Files and Locations

- **Script**: `data_preparation.py`
- **Utilities**: `utils.py` (S3Handler class)
- **Local cache**: `./data_cache/`
- **S3 bucket**: `s3://reasonforge-datasets-dev-071909720457`
- **Metadata**: `./data_cache/upload_metadata_*.json`

### Related Documentation

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training pipeline
- [README.md](README.md) - Project overview
