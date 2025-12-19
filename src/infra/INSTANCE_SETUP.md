# Manual EC2 Training Instance Setup Guide

This guide walks you through creating a g5.xlarge EC2 instance with A10G GPU for training Qwen3-4B using Unsloth.

## Prerequisites

Before starting, ensure you have:
- Deployed S3 buckets and IAM roles using `./deploy.sh`
- AWS CLI configured
- SSH key pair created (or create one during instance launch)

## Step-by-Step Instance Creation

### 1. Navigate to EC2 Console

1. Open [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
2. Select your region (e.g., **us-east-1**)
3. Click **Launch Instance**

### 2. Configure Instance

#### Name and Tags
```
Name: reasonforge-training-dev
```

Add tags:
- Key: `Project` | Value: `ReasonForge`
- Key: `Purpose` | Value: `LLM Training`
- Key: `Model` | Value: `Qwen3-4B`

#### Application and OS Images (AMI)

1. Click **Browse more AMIs**
2. Select **AWS Marketplace AMIs** tab
3. Search for: `Deep Learning AMI GPU PyTorch`
4. Select: **Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 20.04)**
   - AMI includes: PyTorch 2.1, CUDA 12.1, cuDNN 8
5. Click **Select**

#### Instance Type

1. Click on **Instance type** dropdown
2. Search for: `g5.xlarge`
3. Select: **g5.xlarge**
   - 1x NVIDIA A10G GPU (24 GB GPU memory)
   - 4 vCPUs
   - 16 GB RAM
   - Up to 10 Gbps network bandwidth

**Cost**: ~$1.006/hr (On-Demand) or ~$0.30/hr (Spot)

To use **Spot Instance** for cost savings:
1. Expand **Advanced details** at bottom
2. Under **Purchasing option**, check **Request Spot Instances**
3. Leave **Maximum price** blank (use default)

#### Key Pair

Option 1: Use existing key pair
- Select your existing key pair from dropdown

Option 2: Create new key pair
1. Click **Create new key pair**
2. Name: `reasonforge-training`
3. Type: **RSA**
4. Format: **.pem** (for Mac/Linux) or **.ppk** (for Windows/PuTTY)
5. Click **Create key pair**
6. Save the `.pem` file securely
7. Set permissions (Mac/Linux):
   ```bash
   chmod 400 ~/.ssh/reasonforge-training.pem
   ```

### 3. Network Settings

#### VPC and Subnet
- Use default VPC
- Select any subnet (preferably one with good availability)
- **Auto-assign public IP**: Enable

#### Security Group

Create new security group:
1. Name: `reasonforge-training-sg`
2. Description: `Security group for ReasonForge training instance`

Add inbound rules:

| Type | Protocol | Port Range | Source | Description |
|------|----------|------------|--------|-------------|
| SSH | TCP | 22 | My IP | SSH access |
| Custom TCP | TCP | 8888 | My IP | Jupyter Lab |
| Custom TCP | TCP | 6006 | My IP | TensorBoard |

**Important**: Use "My IP" for source to restrict access to your IP address only.

### 4. Configure Storage

1. Root volume (default `/dev/sda1`):
   - Size: **200 GiB** (minimum recommended)
   - Volume type: **gp3**
   - IOPS: **3000**
   - Throughput: **125 MB/s**
   - Delete on termination: **Yes**
   - Encrypted: **Yes**

### 5. Advanced Details

#### IAM Instance Profile
1. Expand **Advanced details**
2. Under **IAM instance profile**, select:
   ```
   reasonforge-training-instance-profile-dev
   ```
   This gives the instance access to S3 buckets.

#### User Data (Optional - Recommended)

Paste this script in **User data** field to auto-configure the instance:

```bash
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Set environment
export HOME=/home/ubuntu
export USER=ubuntu

# Activate conda pytorch environment
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Upgrade pip
pip install --upgrade pip

# Install Unsloth and dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Install additional tools
pip install datasets wandb tensorboard jupyter jupyterlab

# Create directory structure
sudo -u ubuntu mkdir -p /home/ubuntu/reasonforge/{data,models,adapters,checkpoints,logs}

# Setup Jupyter config
sudo -u ubuntu jupyter notebook --generate-config
cat >> /home/ubuntu/.jupyter/jupyter_notebook_config.py <<EOF
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
EOF

# Create startup script for Jupyter
cat > /home/ubuntu/start_jupyter.sh <<'SCRIPT'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch
cd /home/ubuntu/reasonforge
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser > /home/ubuntu/jupyter.log 2>&1 &
echo "Jupyter Lab started. Access at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888"
echo "Check /home/ubuntu/jupyter.log for token"
SCRIPT
chmod +x /home/ubuntu/start_jupyter.sh
chown ubuntu:ubuntu /home/ubuntu/start_jupyter.sh

echo "Setup complete!" > /home/ubuntu/setup_complete.txt
```

### 6. Summary and Launch

1. Review the **Summary** panel on right
2. Verify:
   - Instance type: g5.xlarge
   - AMI: Deep Learning AMI GPU PyTorch
   - Storage: 200 GiB gp3
   - Security group allows SSH, Jupyter, TensorBoard
   - IAM role attached
3. Click **Launch instance**

### 7. Wait for Instance to Start

1. Click **View all instances**
2. Wait for **Instance state** to show **Running**
3. Wait for **Status checks** to show **2/2 checks passed** (takes 2-5 minutes)

## Connecting to Your Instance

### Get Instance Information

```bash
# Get instance public IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=reasonforge-training-dev" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

Or from EC2 Console:
1. Select your instance
2. Copy **Public IPv4 address** from details

### SSH Connection

```bash
# Replace with your actual IP
INSTANCE_IP=<YOUR_INSTANCE_IP>

# Connect via SSH
ssh -i ~/.ssh/reasonforge-training.pem ubuntu@${INSTANCE_IP}
```

### Verify GPU

Once connected, verify the GPU:

```bash
# Check GPU
nvidia-smi

# Should show:
# - 1x NVIDIA A10G
# - 24 GB GPU memory
# - CUDA Version 12.1
```

### Activate Conda Environment

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Verify PyTorch with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Setting Up Your Training Environment

### 1. Install Unsloth (if not done via User Data)

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

### 2. Install Additional Tools

```bash
pip install datasets wandb tensorboard jupyter jupyterlab
```

### 3. Create Directory Structure

```bash
mkdir -p ~/reasonforge/{data,models,adapters,checkpoints,logs}
cd ~/reasonforge
```

### 4. Download Dataset from S3

```bash
# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Sync dataset from S3
aws s3 sync s3://reasonforge-datasets-dev-${ACCOUNT_ID}/turing-open-reasoning/ ~/reasonforge/data/
```

### 5. Clone Your Training Code (if applicable)

```bash
cd ~/reasonforge
# git clone your training repo or copy training scripts
```

## Starting Jupyter Lab

### Option 1: Using Startup Script (if User Data was used)

```bash
/home/ubuntu/start_jupyter.sh
```

### Option 2: Manual Start

```bash
cd ~/reasonforge
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser > jupyter.log 2>&1 &
```

### Access Jupyter Lab

1. Get the token from logs:
   ```bash
   grep "token=" ~/jupyter.log
   ```

2. Open in browser:
   ```
   http://<INSTANCE_IP>:8888
   ```

3. Enter the token when prompted

## Running Training

### Basic Training Script Example

Create `train.py`:

```python
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load dataset
dataset = load_dataset("json", data_files="data/turing-open-reasoning.jsonl")

# Training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="checkpoints",
    ),
)

# Train
trainer.train()

# Save model
model.save_pretrained("adapters/qwen3-4b-lora")
tokenizer.save_pretrained("adapters/qwen3-4b-lora")
```

### Run Training

```bash
python train.py
```

### Monitor with TensorBoard

In another SSH session:

```bash
cd ~/reasonforge
tensorboard --logdir=./checkpoints --host=0.0.0.0 --port=6006
```

Access at: `http://<INSTANCE_IP>:6006`

## Upload Results to S3

After training completes:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Upload checkpoints
aws s3 sync ~/reasonforge/checkpoints/ s3://reasonforge-models-dev-${ACCOUNT_ID}/checkpoints/

# Upload LoRA adapters
aws s3 sync ~/reasonforge/adapters/ s3://reasonforge-adapters-dev-${ACCOUNT_ID}/qwen3-4b-lora/

# Upload logs
aws s3 sync ~/reasonforge/logs/ s3://reasonforge-models-dev-${ACCOUNT_ID}/logs/
```

## Cost Management

### Stop Instance When Not Training

```bash
# From your local machine
aws ec2 stop-instances --instance-ids <INSTANCE_ID>
```

Or in AWS Console:
1. Select instance
2. Instance state → Stop instance

**Note**: You still pay for EBS storage when stopped (~$20/month for 200 GB gp3)

### Terminate Instance When Done

```bash
# From your local machine
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

Or in AWS Console:
1. Select instance
2. Instance state → Terminate instance

**Warning**: This permanently deletes the instance and all data on it. Make sure you've uploaded everything to S3 first!

## Troubleshooting

### Can't Connect via SSH

1. Check security group allows your IP on port 22
2. Verify instance is running
3. Check you're using correct key pair
4. Verify public IP hasn't changed (if instance was stopped/started)

### GPU Not Available

```bash
# Check if GPU is visible
nvidia-smi

# If not visible, reboot instance
sudo reboot
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up
conda clean --all -y
pip cache purge
rm -rf ~/.cache/*
```

To increase disk:
1. Stop instance
2. Modify volume size in EC2 Console
3. Start instance
4. Resize filesystem:
   ```bash
   sudo growpart /dev/nvme0n1 1
   sudo resize2fs /dev/nvme0n1p1
   ```

### Conda Environment Issues

```bash
# Reset conda
source /opt/conda/etc/profile.d/conda.sh
conda deactivate
conda activate pytorch
```

## Next Steps

1. Download and preprocess the TuringEnterprises/Turing-Open-Reasoning dataset
2. Set up Weights & Biases for experiment tracking
3. Implement training script with Unsloth
4. Fine-tune Qwen3-4B model
5. Upload trained adapters to S3
6. Deploy to EKS with Ray Serve and vLLM

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
- [g5 Instance Pricing](https://aws.amazon.com/ec2/instance-types/g5/pricing/)
- [EC2 Spot Instances](https://aws.amazon.com/ec2/spot/)
