"""
LoRA Fine-tuning Script
=======================

This script handles the complete LoRA fine-tuning pipeline:
1. Loading pre-trained model with Unsloth optimizations
2. Loading processed dataset from S3
3. Configuring LoRA parameters
4. Training with monitoring
5. Saving adapter to S3 and HuggingFace Hub

Model: unsloth/Qwen3-4B
Optimization: Unsloth for 2x faster training with reduced memory
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import time

import torch
from datasets import load_from_disk, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer

from utils import (
    S3Handler,
    MetricsLogger,
    get_gpu_info,
    print_gpu_memory,
    format_time,
    save_training_config,
    print_training_info
)


class LoRATrainer:
    """Handles LoRA fine-tuning with Unsloth optimizations."""

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-4B",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[list] = None,
        local_cache_dir: str = "./model_cache",
    ):
        """
        Initialize LoRA trainer.

        Args:
            model_name: HuggingFace model identifier
            max_seq_length: Maximum sequence length for training
            load_in_4bit: Use 4-bit quantization for memory efficiency
            lora_r: LoRA rank (higher = more parameters but better quality)
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout for LoRA layers
            target_modules: Modules to apply LoRA to (None = auto-detect)
            local_cache_dir: Directory for caching models
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # LoRA configuration
        self.lora_config = {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "use_rslora": False,
            "use_dora": False,
        }

        self.model = None
        self.tokenizer = None
        self.s3_handler = S3Handler()

        print(f"✓ Initialized LoRATrainer")
        print(f"  Model: {model_name}")
        print(f"  Max Sequence Length: {max_seq_length}")
        print(f"  4-bit Quantization: {load_in_4bit}")
        print(f"  LoRA Rank: {lora_r}")

    def load_model(self):
        """Load model with Unsloth optimizations and LoRA configuration."""
        print(f"\n🔄 Loading model: {self.model_name}")
        print_gpu_memory()

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.load_in_4bit,
        )

        print(f"✓ Base model loaded")
        print_gpu_memory()

        # Add LoRA adapters
        print(f"\n🔧 Adding LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            target_modules=self.lora_config["target_modules"],
            bias=self.lora_config["bias"],
            use_gradient_checkpointing=self.lora_config["use_gradient_checkpointing"],
            use_rslora=self.lora_config["use_rslora"],
        )

        print(f"✓ LoRA adapters added")
        print_gpu_memory()

        # Print trainable parameters
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_percent = 100 * trainable_params / all_params

        print(f"\n📊 Model Parameters:")
        print(f"  - Total: {all_params:,}")
        print(f"  - Trainable: {trainable_params:,}")
        print(f"  - Trainable %: {trainable_percent:.2f}%")

    def load_dataset(
        self,
        dataset_path: str,
        from_s3: bool = True
    ) -> DatasetDict:
        """
        Load dataset from local path or S3.

        Args:
            dataset_path: Path to dataset (local or S3 URI)
            from_s3: Whether to download from S3 first

        Returns:
            Loaded DatasetDict
        """
        print(f"\n📥 Loading dataset...")

        if from_s3 and dataset_path.startswith("s3://"):
            # Download from S3
            local_path = self.local_cache_dir / "dataset"
            dataset_path = self.s3_handler.download_directory(
                dataset_path,
                local_path
            )
        else:
            dataset_path = Path(dataset_path)

        print(f"📂 Loading from: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))

        if isinstance(dataset, DatasetDict):
            print(f"✓ Loaded dataset with splits:")
            for split_name, split_data in dataset.items():
                print(f"  - {split_name}: {len(split_data):,} examples")
        else:
            print(f"✓ Loaded dataset: {len(dataset):,} examples")

        return dataset

    def train(
        self,
        dataset: DatasetDict,
        output_dir: str = "./checkpoints",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 5,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100,
        max_grad_norm: float = 0.3,
        weight_decay: float = 0.01,
        optim: str = "adamw_8bit",
        lr_scheduler_type: str = "linear",
        seed: int = 42,
    ) -> str:
        """
        Train the model with LoRA.

        Args:
            dataset: DatasetDict with 'train' and 'validation' splits
            output_dir: Directory for checkpoints and outputs
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            logging_steps: Log metrics every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            max_grad_norm: Maximum gradient norm for clipping
            weight_decay: Weight decay for regularization
            optim: Optimizer to use
            lr_scheduler_type: Learning rate scheduler type
            seed: Random seed

        Returns:
            Path to output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save training configuration
        training_config = {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "lora_config": self.lora_config,
            "timestamp": datetime.now().isoformat(),
        }

        save_training_config(
            training_config,
            output_dir / "training_config.json"
        )

        # Print training info
        print_training_info(
            model_name=self.model_name,
            dataset_info={
                "train": len(dataset["train"]),
                "validation": len(dataset["validation"])
            },
            config=training_config
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",  # Updated from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=optim,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=max_grad_norm,
            seed=seed,
            report_to="none",  # Disable wandb/tensorboard for simplicity
            logging_dir=str(output_dir / "logs"),
            save_total_limit=3,  # Keep only last 3 checkpoints
        )

        # Initialize trainer
        print(f"\n🏋️  Initializing SFTTrainer...")

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            dataset_text_field="text",  # Field containing formatted text
            max_seq_length=self.max_seq_length,
            args=training_args,
            packing=False,  # Don't pack multiple examples in one sequence
        )

        print(f"✓ Trainer initialized")

        # Start training
        print(f"\n🚀 Starting training...")
        print("=" * 70)

        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time

        print("=" * 70)
        print(f"✅ Training completed in {format_time(training_time)}")

        # Print final metrics
        metrics = train_result.metrics
        print(f"\n📊 Final Training Metrics:")
        print(f"  - Training Loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"  - Training Steps: {metrics.get('train_steps', 'N/A')}")
        print(f"  - Training Time: {format_time(training_time)}")

        # Save final model
        final_model_dir = output_dir / "final_model"
        print(f"\n💾 Saving final model to {final_model_dir}...")

        trainer.save_model(str(final_model_dir))
        self.tokenizer.save_pretrained(str(final_model_dir))

        # Save metrics
        metrics_file = output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Model and metrics saved")
        print_gpu_memory()

        return str(output_dir)

    def save_adapter_to_s3(
        self,
        adapter_path: Path,
        s3_bucket: str,
        s3_prefix: str = "adapters"
    ) -> str:
        """
        Upload LoRA adapter to S3.

        Args:
            adapter_path: Local path to adapter
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix for adapters

        Returns:
            S3 URI of uploaded adapter
        """
        print(f"\n☁️  Uploading adapter to S3...")

        s3_uri = self.s3_handler.upload_directory(
            local_path=adapter_path,
            bucket=s3_bucket,
            s3_prefix=s3_prefix,
            include_timestamp=True
        )

        print(f"✓ Adapter uploaded to: {s3_uri}")
        return s3_uri

    def push_to_hub(
        self,
        adapter_path: Path,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False
    ):
        """
        Push LoRA adapter to HuggingFace Hub.

        Args:
            adapter_path: Local path to adapter
            repo_id: HuggingFace repository ID (username/repo-name)
            token: HuggingFace API token (optional, uses HF_TOKEN env var)
            private: Whether to make the repo private
        """
        print(f"\n🤗 Pushing adapter to HuggingFace Hub: {repo_id}")

        from huggingface_hub import HfApi

        api = HfApi()
        token = token or os.getenv("HF_TOKEN")

        try:
            # Upload folder to hub
            api.upload_folder(
                folder_path=str(adapter_path),
                repo_id=repo_id,
                repo_type="model",
                token=token,
                create_pr=False,
            )

            print(f"✓ Adapter pushed to: https://huggingface.co/{repo_id}")

        except Exception as e:
            print(f"✗ Failed to push to hub: {e}")
            raise


def main():
    """Main execution flow for LoRA training."""

    parser = argparse.ArgumentParser(description="LoRA Fine-tuning with Unsloth")

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/Qwen3-4B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset (local or S3 URI)"
    )
    parser.add_argument(
        "--from-s3",
        action="store_true",
        help="Download dataset from S3"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps"
    )

    # S3 and Hub arguments
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="reasonforge-adapters-dev-071909720457",
        help="S3 bucket for adapter storage"
    )
    parser.add_argument(
        "--skip-s3-upload",
        action="store_true",
        help="Skip uploading adapter to S3"
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default="ashutosh2211/qwen3-4b-reasonforge",
        help="Push adapter to HuggingFace Hub (provide repo-id)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token"
    )

    args = parser.parse_args()

    # Print GPU info
    gpu_info = get_gpu_info()
    print("\n" + "=" * 70)
    print("🎮 GPU INFORMATION")
    print("=" * 70)
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Initialize trainer
    trainer = LoRATrainer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Load model
    trainer.load_model()

    # Load dataset
    dataset = trainer.load_dataset(
        dataset_path=args.dataset_path,
        from_s3=args.from_s3
    )

    # Train
    output_dir = trainer.train(
        dataset=dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
    )

    final_model_path = Path(output_dir) / "final_model"

    # Upload to S3
    if not args.skip_s3_upload:
        s3_uri = trainer.save_adapter_to_s3(
            adapter_path=final_model_path,
            s3_bucket=args.s3_bucket
        )

        print(f"\n📍 Adapter saved to S3: {s3_uri}")

    # Push to HuggingFace Hub
    if args.push_to_hub:
        trainer.push_to_hub(
            adapter_path=final_model_path,
            repo_id=args.push_to_hub,
            token=args.hf_token
        )

    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"📁 Local checkpoint: {output_dir}")
    if not args.skip_s3_upload:
        print(f"☁️  S3 URI: {s3_uri}")
    if args.push_to_hub:
        print(f"🤗 HuggingFace Hub: https://huggingface.co/{args.push_to_hub}")
    print("=" * 70)


if __name__ == "__main__":
    main()
