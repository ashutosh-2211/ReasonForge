"""
Utility Functions for LoRA Fine-tuning Pipeline
================================================

Common utilities used across data preparation, training, and evaluation.
"""

import os
import json
import boto3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch


class S3Handler:
    """Handles S3 operations for models, datasets, and adapters."""

    def __init__(self, region: str = "us-east-1"):
        """
        Initialize S3 handler.

        Args:
            region: AWS region for S3 operations
        """
        self.s3_client = boto3.client('s3', region_name=region)
        self.region = region

    def upload_directory(
        self,
        local_path: Path,
        bucket: str,
        s3_prefix: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Upload entire directory to S3.

        Args:
            local_path: Local directory path
            bucket: S3 bucket name
            s3_prefix: S3 prefix (folder path)
            include_timestamp: Whether to append timestamp to S3 path

        Returns:
            S3 URI of uploaded directory
        """
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key_prefix = f"{s3_prefix}/{local_path.name}_{timestamp}"
        else:
            s3_key_prefix = f"{s3_prefix}/{local_path.name}"

        print(f"📤 Uploading {local_path} to s3://{bucket}/{s3_key_prefix}")

        uploaded_files = []
        for local_file in local_path.rglob("*"):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_path)
                s3_key = f"{s3_key_prefix}/{relative_path}"

                try:
                    self.s3_client.upload_file(
                        str(local_file),
                        bucket,
                        s3_key
                    )
                    uploaded_files.append(s3_key)
                except Exception as e:
                    print(f"✗ Failed to upload {relative_path}: {e}")
                    raise

        s3_uri = f"s3://{bucket}/{s3_key_prefix}"
        print(f"✓ Uploaded {len(uploaded_files)} files to {s3_uri}")
        return s3_uri

    def download_directory(
        self,
        s3_uri: str,
        local_path: Path
    ) -> Path:
        """
        Download entire directory from S3.

        Args:
            s3_uri: S3 URI (s3://bucket/prefix)
            local_path: Local destination path

        Returns:
            Local path to downloaded directory
        """
        # Parse S3 URI
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        parts = s3_uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        print(f"📥 Downloading from {s3_uri} to {local_path}")

        local_path.mkdir(parents=True, exist_ok=True)

        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        downloaded_files = 0
        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                s3_key = obj['Key']
                relative_path = Path(s3_key).relative_to(prefix)
                local_file = local_path / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    self.s3_client.download_file(bucket, s3_key, str(local_file))
                    downloaded_files += 1
                except Exception as e:
                    print(f"✗ Failed to download {s3_key}: {e}")
                    raise

        print(f"✓ Downloaded {downloaded_files} files")
        return local_path

    def upload_file(self, local_file: Path, bucket: str, s3_key: str) -> str:
        """
        Upload single file to S3.

        Args:
            local_file: Local file path
            bucket: S3 bucket name
            s3_key: S3 object key

        Returns:
            S3 URI of uploaded file
        """
        self.s3_client.upload_file(str(local_file), bucket, s3_key)
        return f"s3://{bucket}/{s3_key}"


class MetricsLogger:
    """Logs training metrics and results."""

    def __init__(self, log_dir: Path, experiment_name: str):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.metrics = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"

    def log_metric(self, step: int, metrics: Dict[str, Any]):
        """
        Log metrics for a training step.

        Args:
            step: Training step number
            metrics: Dictionary of metric names and values
        """
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)

        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log metrics for a training epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names and values
        """
        self.log_metric(step=epoch, metrics={"epoch": epoch, **metrics})

    def save_summary(self) -> Path:
        """
        Save summary of all metrics.

        Returns:
            Path to summary file
        """
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"

        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(self.metrics),
            "metrics": self.metrics
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary_file

    def get_best_metric(self, metric_name: str, mode: str = "min") -> Dict:
        """
        Get best value for a specific metric.

        Args:
            metric_name: Name of metric to find best value for
            mode: 'min' or 'max' - whether lower or higher is better

        Returns:
            Dictionary with best metric value and step
        """
        if not self.metrics:
            return {}

        valid_metrics = [m for m in self.metrics if metric_name in m]
        if not valid_metrics:
            return {}

        if mode == "min":
            best = min(valid_metrics, key=lambda x: x[metric_name])
        else:
            best = max(valid_metrics, key=lambda x: x[metric_name])

        return best


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information and memory stats.

    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    gpu_info = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
        "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(0) / 1e9,
    }

    return gpu_info


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"🎮 GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
    else:
        print("⚠️  CUDA not available")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_training_config(config: Dict[str, Any], save_path: Path):
    """
    Save training configuration to file.

    Args:
        config: Training configuration dictionary
        save_path: Path to save configuration
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"💾 Configuration saved to {save_path}")


def load_training_config(config_path: Path) -> Dict[str, Any]:
    """
    Load training configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Training configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def print_training_info(
    model_name: str,
    dataset_info: Dict[str, int],
    config: Dict[str, Any]
):
    """
    Print formatted training information.

    Args:
        model_name: Name of the model
        dataset_info: Dataset statistics (train/val sizes)
        config: Training configuration
    """
    print("\n" + "=" * 70)
    print("🚀 LORA FINE-TUNING SETUP")
    print("=" * 70)
    print(f"\n📦 Model: {model_name}")
    print(f"\n📊 Dataset:")
    print(f"  - Training samples: {dataset_info.get('train', 0):,}")
    print(f"  - Validation samples: {dataset_info.get('validation', 0):,}")
    print(f"\n⚙️  Training Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    gpu_info = get_gpu_info()
    if gpu_info.get("cuda_available"):
        print(f"\n🎮 GPU: {gpu_info['device_name']}")
        print(f"  - Memory: {gpu_info['memory_allocated_gb']:.2f}GB allocated")
    else:
        print(f"\n⚠️  No GPU available - training will be slow!")

    print("=" * 70 + "\n")
