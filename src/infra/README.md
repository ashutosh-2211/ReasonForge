# ReasonForge Infrastructure

This directory contains infrastructure as code (IaC) for deploying the ReasonForge training and serving environment.

## Overview

The infrastructure consists of:

1. **S3 Buckets** - Storage for models, adapters, and datasets
2. **IAM Roles** - Security and access policies
3. **EC2 Training Instance** - g5.xlarge instance with A10G GPU for training Qwen3-4B

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ReasonForge Infrastructure              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌─────────────────────────────┐  │
│  │              │         │  EC2 Training Instance      │  │
│  │  S3 Buckets  │◄────────│  - g5.xlarge (A10G GPU)    │  │
│  │              │         │  - Deep Learning AMI        │  │
│  │  • Models    │         │  - Unsloth + PyTorch        │  │
│  │  • Adapters  │         │  - Jupyter Lab              │  │
│  │  • Datasets  │         │  - TensorBoard              │  │
│  │              │         └─────────────────────────────┘  │
│  └──────────────┘                      │                    │
│                                        │                    │
│                            ┌───────────▼─────────────┐      │
│                            │    IAM Instance Role    │      │
│                            │  - S3 Access            │      │
│                            │  - CloudWatch Logs      │      │
│                            │  - ECR Access           │      │
│                            └─────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/infra/
├── cloudformation/
│   ├── 01-s3-buckets.yaml          # S3 buckets for storage
│   ├── 02-iam-roles.yaml           # IAM roles and policies
│   └── 03-training-instance.yaml   # EC2 training instance
├── config.yaml                     # Configuration reference
├── deploy.sh                       # Deployment script
├── destroy.sh                      # Cleanup script
└── README.md                       # This file
```

## Prerequisites

1. **AWS CLI** installed and configured
   ```bash
   aws configure
   ```

2. **EC2 Key Pair** for SSH access
   ```bash
   # Create a new key pair
   aws ec2 create-key-pair \
     --key-name reasonforge-training \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/reasonforge-training.pem

   chmod 400 ~/.ssh/reasonforge-training.pem
   ```

3. **Permissions** - Your AWS IAM user/role needs:
   - CloudFormation full access
   - S3 full access
   - EC2 full access
   - IAM role creation

## Quick Start

### 1. Deploy S3 Buckets and IAM Roles Only

This creates the storage infrastructure without launching an EC2 instance:

```bash
cd src/infra
./deploy.sh
```

This will create:
- `reasonforge-models-dev-{ACCOUNT_ID}`
- `reasonforge-adapters-dev-{ACCOUNT_ID}`
- `reasonforge-datasets-dev-{ACCOUNT_ID}`
- IAM roles for training instances

### 2. Deploy Everything (Including Training Instance)

```bash
./deploy.sh --key-pair reasonforge-training --ssh-cidr YOUR_IP/32
```

Replace `YOUR_IP` with your actual IP address for secure SSH access.

### 3. Deploy with Spot Instance (Cost Savings)

```bash
./deploy.sh --key-pair reasonforge-training --spot
```

**Note:** Spot instances can be interrupted by AWS with 2-minute notice.

## Deployment Options

```bash
# Deploy to a different region
./deploy.sh --region us-west-2 --key-pair my-key

# Deploy to staging environment
./deploy.sh --environment staging --key-pair my-key

# Deploy only the training instance (buckets and IAM must exist)
./deploy.sh --instance-only --key-pair my-key

# Use spot instance for cost savings
./deploy.sh --key-pair my-key --spot

# Restrict SSH to your IP
./deploy.sh --key-pair my-key --ssh-cidr 203.0.113.0/32

# View all options
./deploy.sh --help
```

## Accessing the Training Instance

After deployment, you'll receive connection information:

### SSH Access

```bash
ssh -i ~/.ssh/reasonforge-training.pem ubuntu@<INSTANCE_IP>
```

### Jupyter Lab

```bash
# On the instance, start Jupyter Lab
/home/ubuntu/start_jupyter.sh

# Access from browser
http://<INSTANCE_IP>:8888
```

### TensorBoard

```bash
# On the instance
cd /home/ubuntu/reasonforge
tensorboard --logdir=./logs --host=0.0.0.0

# Access from browser
http://<INSTANCE_IP>:6006
```

## Working with S3 Buckets

### Upload Dataset

```bash
# Get bucket name (replace ACCOUNT_ID)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
DATASET_BUCKET="reasonforge-datasets-dev-${ACCOUNT_ID}"

# Upload from local
aws s3 cp ./local-dataset.jsonl s3://${DATASET_BUCKET}/turing-open-reasoning/

# Or download directly on instance
ssh ubuntu@<INSTANCE_IP>
aws s3 sync s3://${DATASET_BUCKET}/turing-open-reasoning/ /home/ubuntu/reasonforge/data/
```

### Upload/Download Models

```bash
MODELS_BUCKET="reasonforge-models-dev-${ACCOUNT_ID}"

# Upload fine-tuned model
aws s3 sync ./model_output s3://${MODELS_BUCKET}/qwen3-4b-finetuned/

# Download on instance
aws s3 sync s3://${MODELS_BUCKET}/qwen3-4b-base/ /home/ubuntu/reasonforge/models/
```

### Upload/Download Adapters

```bash
ADAPTERS_BUCKET="reasonforge-adapters-dev-${ACCOUNT_ID}"

# Upload LoRA adapters
aws s3 sync ./lora_adapters s3://${ADAPTERS_BUCKET}/qwen3-4b-lora/
```

## Training Workflow

1. **SSH into the instance**
   ```bash
   ssh -i ~/.ssh/reasonforge-training.pem ubuntu@<INSTANCE_IP>
   ```

2. **Activate conda environment**
   ```bash
   source /opt/conda/etc/profile.d/conda.sh
   conda activate pytorch
   ```

3. **Download dataset from S3**
   ```bash
   ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
   aws s3 sync s3://reasonforge-datasets-dev-${ACCOUNT_ID}/turing-open-reasoning/ \
     /home/ubuntu/reasonforge/data/
   ```

4. **Run training**
   ```bash
   cd /home/ubuntu/reasonforge
   # Your training script here
   python train.py
   ```

5. **Upload trained model/adapters**
   ```bash
   aws s3 sync ./checkpoints/ s3://reasonforge-models-dev-${ACCOUNT_ID}/checkpoints/
   aws s3 sync ./adapters/ s3://reasonforge-adapters-dev-${ACCOUNT_ID}/run-001/
   ```

## Instance Setup

The training instance comes pre-configured with:

- **Deep Learning AMI** (Ubuntu 20.04, PyTorch 2.1, CUDA 12.1)
- **Unsloth** for efficient fine-tuning
- **Python packages**: transformers, datasets, accelerate, bitsandbytes, peft, trl
- **Tools**: Jupyter Lab, TensorBoard, AWS CLI
- **Directory structure**: `/home/ubuntu/reasonforge/{data,models,adapters,checkpoints,logs}`

## Cost Optimization

### Instance Costs (us-east-1)

| Instance Type | vCPUs | GPU Memory | RAM | On-Demand | Spot (avg) |
|---------------|-------|------------|-----|-----------|------------|
| g5.xlarge     | 4     | 24 GB (A10G)| 16 GB | $1.006/hr | ~$0.30/hr |
| g5.2xlarge    | 8     | 24 GB (A10G)| 32 GB | $1.212/hr | ~$0.36/hr |

### Tips

1. **Use Spot Instances** - Save ~70% with `--spot` flag
2. **Stop when not training** - Don't leave instances running idle
3. **Use Intelligent-Tiering** - S3 buckets auto-configured for cost optimization
4. **Monitor usage** - Set up billing alerts in AWS Console

## Cleanup

### Destroy Training Instance Only

```bash
./destroy.sh --instance-only
```

This keeps S3 buckets and IAM roles intact.

### Destroy Everything

```bash
./destroy.sh --all
```

**WARNING:** This will:
- Delete the EC2 instance
- Delete IAM roles
- **Permanently delete all S3 buckets and their contents**

## Troubleshooting

### Stack Creation Fails

1. Check AWS region supports g5.xlarge instances
2. Verify you have sufficient EC2 quotas
3. Ensure key pair exists in the target region

### Cannot Connect via SSH

1. Check security group allows your IP
2. Verify instance is in "running" state
3. Check you're using the correct key pair

### Instance Out of Disk Space

The root volume is 200 GB by default. To increase:

```bash
# Edit cloudformation/03-training-instance.yaml
# Change RootVolumeSize default value
# Redeploy: ./deploy.sh --instance-only --key-pair YOUR_KEY
```

## Manual Deployment (Alternative)

If you prefer to deploy stacks individually:

```bash
# 1. S3 Buckets
aws cloudformation create-stack \
  --stack-name reasonforge-s3-buckets-dev \
  --template-body file://cloudformation/01-s3-buckets.yaml \
  --parameters ParameterKey=ProjectName,ParameterValue=reasonforge \
               ParameterKey=Environment,ParameterValue=dev

# 2. IAM Roles
aws cloudformation create-stack \
  --stack-name reasonforge-iam-roles-dev \
  --template-body file://cloudformation/02-iam-roles.yaml \
  --parameters ParameterKey=ProjectName,ParameterValue=reasonforge \
               ParameterKey=Environment,ParameterValue=dev \
  --capabilities CAPABILITY_NAMED_IAM

# 3. Training Instance
aws cloudformation create-stack \
  --stack-name reasonforge-training-instance-dev \
  --template-body file://cloudformation/03-training-instance.yaml \
  --parameters ParameterKey=ProjectName,ParameterValue=reasonforge \
               ParameterKey=Environment,ParameterValue=dev \
               ParameterKey=KeyPairName,ParameterValue=YOUR_KEY_PAIR \
               ParameterKey=AllowedSSHCIDR,ParameterValue=YOUR_IP/32 \
  --capabilities CAPABILITY_NAMED_IAM
```

## Next Steps

1. **Data Collection**: Download TuringEnterprises/Turing-Open-Reasoning dataset
2. **Training**: Use Unsloth to fine-tune unsloth/Qwen3-4B
3. **Serving**: Deploy to EKS with Ray Serve and vLLM (see `eks/` directory)

## Support

For issues or questions:
- Check CloudFormation events in AWS Console
- Review `/var/log/cloud-init-output.log` on the instance
- Verify IAM permissions
